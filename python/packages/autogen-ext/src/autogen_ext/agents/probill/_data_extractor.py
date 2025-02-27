from typing import Any, Dict, List, Sequence
import asyncio
import base64
import json
import logging
import traceback
from io import BytesIO

from autogen_agentchat.agents import BaseChatAgent, AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import ChatMessage, TextMessage, MultiModalMessage
from autogen_core import CancellationToken, Component, ComponentModel
from autogen_core.models import ChatCompletionClient, LLMMessage, SystemMessage, UserMessage
from autogen_core.tools import BaseTool, FunctionTool
from PIL import Image
from pydantic import BaseModel

# Updated configuration schema with a mandatory healthcare_provider_finder tool.
class DataExtractorAgentConfig(BaseModel):
    name: str
    model_client: ComponentModel
    downloads_folder: str | None = None
    description: str | None = None
    debug_dir: str | None = None
    headless: bool = True
    animate_actions: bool = False
    to_save_screenshots: bool = False
    use_ocr: bool = False
    browser_channel: str | None = None
    browser_data_dir: str | None = None
    to_resize_viewport: bool = True



def dummy_healthcare_provider_finder() -> str:
    return {"name":"dummy_healthcare_provider"}

# Create calculator tool
_healthcare_provider_finder = FunctionTool(
    name="healthcare_provider_finder",
    description="A sdummy healthcare_provider_finder tool.",
    func=dummy_healthcare_provider_finder,
    global_imports=[],
)

class DataExtractor(BaseChatAgent, Component[DataExtractorAgentConfig]):
    """
    DataExtractorAgent uses a fine-tuned vLLM to extract structured data from scanned face sheets.
    It identifies provider-specific markers and, using the healthcare_provider_finder tool, retrieves the appropriate
    extraction template. Then it extracts essential fields (e.g., patient name, insurance ID, date of birth) and outputs the result in JSON.
    """
    component_type = "agent"
    component_config_schema = DataExtractorAgentConfig
    component_provider_override = "autogenstudio.gallery.agents.probill.DataExtractorAgent"
    
    def __init__(
        self,
        name: str,
        model_client: ChatCompletionClient,
        description: str | None = None,
        downloads_folder: str | None = None,
        debug_dir: str | None = None,
        headless: bool = True,
        animate_actions: bool = False,
        to_save_screenshots: bool = False,
        use_ocr: bool = False,
        browser_channel: str | None = None,
        browser_data_dir: str | None = None,
        to_resize_viewport: bool = True,
    ):
        super().__init__(name, "Data Extractor Agent")
        self._description = description
        self._model_client = model_client
        self.headless = headless
        self.browser_channel = browser_channel
        self.browser_data_dir = browser_data_dir
        self.downloads_folder = downloads_folder
        self.debug_dir = debug_dir
        self.to_save_screenshots = to_save_screenshots
        self.use_ocr = use_ocr
        self.to_resize_viewport = to_resize_viewport
        self.animate_actions = animate_actions
        self.logger = logging.getLogger(f"DataExtractorAgent.{self.name}")
        self._chat_history: List[LLMMessage] = []
        self.did_lazy_init = False  # flag to check if we have initialized the browser

    async def _lazy_init(
        self,
    ) -> None:
        """
        On the first call, we initialize the browser and the page.
        """
        self._last_download = None
        self._prior_metadata_hash = None

        # Create the playwright self
        launch_args: Dict[str, Any] = {"headless": self.headless}
        if self.browser_channel is not None:
            launch_args["channel"] = self.browser_channel

        # Create the context -- are we launching persistent?
        if self._context is None:
            if self.browser_data_dir is None:
                browser = await self._playwright.chromium.launch(**launch_args)
                self._context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0"
                )
            else:
                self._context = await self._playwright.chromium.launch_persistent_context(
                    self.browser_data_dir, **launch_args
                )

        # Create the page
        self._context.set_default_timeout(60000)  # One minute
        self._page = await self._context.new_page()
        assert self._page is not None
        # self._page.route(lambda x: True, self._route_handler)
        self._page.on("download", self._download_handler)
        if self.to_resize_viewport:
            await self._page.set_viewport_size({"width": self.VIEWPORT_WIDTH, "height": self.VIEWPORT_HEIGHT})
        await self._page.add_init_script(
            path=os.path.join(os.path.abspath(os.path.dirname(__file__)), "page_script.js")
        )
        await self._page.goto(self.start_page)
        await self._page.wait_for_load_state()

        # Prepare the debug directory -- which stores the screenshots generated throughout the process
        await self._set_debug_dir(self.debug_dir)
        self.did_lazy_init = True

    async def on_messages(self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken) -> Response:
        """
        Handles incoming messages which are expected to contain the face sheet data.
        The last message is assumed to carry the image input (either a file path or a base64 string).
        """
        try:
            # Retrieve the face sheet input from the last message.
            face_sheet_input = messages[-1].content  
            extracted_data = await self.extract_data(face_sheet_input, cancellation_token)
            response_message = TextMessage(content=json.dumps(extracted_data, indent=4), source=self.name)
            return Response(chat_message=response_message)
        except Exception as e:
            error_message = f"Error in DataExtractorAgent: {str(e)}\n{traceback.format_exc()}"
            self.logger.error(error_message)
            return Response(chat_message=TextMessage(content=error_message, source=self.name))

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        pass

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (MultiModalMessage,)

    async def extract_data(self, face_sheet_input: str, cancellation_token: CancellationToken) -> Dict[str, Any]:
        """
        Processes the face sheet input using the fine-tuned vLLM.
        
        Args:
            face_sheet_input: Either a file path to the face sheet image or a base64-encoded string.
            cancellation_token: Token for cancelling the asynchronous operation.
        
        Returns:
            A dictionary containing the extracted fields.
        """
        # Attempt to load the image from a file path; if that fails, decode as base64.
        try:
            image = Image.open(face_sheet_input)
        except Exception:
            image_data = base64.b64decode(face_sheet_input)
            image = Image.open(BytesIO(image_data))
        
        # Ensure the image is in RGB format.
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Use the healthcare_provider_finder tool to match the provider.
        # The tool is expected to return a provider marker (e.g., "default_provider").
        system_msg = SystemMessage(content="You are a healthcare provider finder tool.")
        user_msg = UserMessage(content="Identify the provider based on the face sheet.", source=self.name)
        provider_response = await self.healthcare_provider_finder.create(
            [system_msg, user_msg], cancellation_token=cancellation_token
        )
        provider_marker = provider_response.content.strip()  # Assuming a plain text response.
        
        # Prepare a prompt for the vLLM to perform extraction.
        prompt = (
            "Extract the following fields from the face sheet: patient name, insurance ID, date of birth, "
            "and provider information."
        )
        if provider_marker in self.template_library:
            template = self.template_library[provider_marker]
            prompt += "\nApply the following extraction template: " + json.dumps(template)
        else:
            prompt += "\nUse default extraction rules."
        prompt += "\n[Image data attached]"
        
        # Prepare the conversation history for the vLLM.
        system_message = SystemMessage(content="You are a data extraction assistant specialized in medical billing face sheets.")
        user_message = UserMessage(content=prompt, source=self.name)
        history = [system_message, user_message]
        
        # Call the model client to extract the data.
        response = await self._model_client.create(history, cancellation_token=cancellation_token)
        try:
            extracted_json = json.loads(response.content)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON output: {str(e)}. Raw response: {response.content}")
        return extracted_json

    def _to_config(self) -> DataExtractorAgentConfig:
        return DataExtractorAgentConfig(
            name = self.name,
            model_client = self._model_client.dump_component(),
            downloads_folder=self.downloads_folder,
            description=self.description,
            debug_dir=self.debug_dir,
            headless=self.headless,
            animate_actions=self.animate_actions,
            to_save_screenshots=self.to_save_screenshots,
            use_ocr=self.use_ocr,
            browser_channel=self.browser_channel,
            browser_data_dir=self.browser_data_dir,
            to_resize_viewport=self.to_resize_viewport,
        )

    @classmethod
    def _from_config(cls, config: DataExtractorAgentConfig) -> "DataExtractorAgent":
        return cls(
            name=config.name,
            model_client=ChatCompletionClient.load_component(config.model_client),
            downloads_folder=config.downloads_folder,
            description=config.description,
            debug_dir=config.debug_dir,
            headless=config.headless,
            animate_actions=config.animate_actions,
            to_save_screenshots=config.to_save_screenshots,
            use_ocr=config.use_ocr,
            browser_channel=config.browser_channel,
            browser_data_dir=config.browser_data_dir,
            to_resize_viewport=config.to_resize_viewport,
        )
