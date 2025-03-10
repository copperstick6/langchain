import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.utils import guard_import

from langchain_community.callbacks.utils import import_pandas

import requests
import logging

logger = logging.getLogger(__name__)

# Define constants

# LLMResult keys
TOKEN_USAGE = "token_usage"
TOTAL_TOKENS = "total_tokens"
PROMPT_TOKENS = "prompt_tokens"
COMPLETION_TOKENS = "completion_tokens"
RUN_ID = "run_id"
MODEL_NAME = "model_name"
GOOD = "good"
BAD = "bad"
NEUTRAL = "neutral"
SUCCESS = "success"
FAILURE = "failure"

# Default values
DEFAULT_MAX_TOKEN = 65536
DEFAULT_MAX_DURATION = 120000

# Fiddler specific constants
PROMPT = "prompt"
RESPONSE = "response"
CONTEXT = "context"
DURATION = "duration"
FEEDBACK = "feedback"
LLM_STATUS = "llm_status"

FEEDBACK_POSSIBLE_VALUES = [GOOD, BAD, NEUTRAL]

# Define a dataset dictionary
_dataset_dict = {
    PROMPT: ["fiddler"] * 10,
    RESPONSE: ["fiddler"] * 10,
    CONTEXT: ["fiddler"] * 10,
    FEEDBACK: ["good"] * 10,
    LLM_STATUS: ["success"] * 10,
    MODEL_NAME: ["fiddler"] * 10,
    RUN_ID: ["123e4567-e89b-12d3-a456-426614174000"] * 10,
    TOTAL_TOKENS: [0, DEFAULT_MAX_TOKEN] * 5,
    PROMPT_TOKENS: [0, DEFAULT_MAX_TOKEN] * 5,
    COMPLETION_TOKENS: [0, DEFAULT_MAX_TOKEN] * 5,
    DURATION: [1, DEFAULT_MAX_DURATION] * 5,
}


def import_fiddler() -> Any:
    """Import the fiddler python package and raise an error if it is not installed."""
    return guard_import("fiddler", pip_name="fiddler-client")


# First, define custom callback handler implementations
class FiddlerCallbackHandler(BaseCallbackHandler):
    def __init__(
        self,
        url: str,
        org: str,
        project: str,
        model: str,
        api_key: str,
    ) -> None:
        """
        Initialize Fiddler callback handler.

        Args:
            url: Fiddler URL (e.g. https://demo.fiddler.ai).
                Make sure to include the protocol (http/https).
            org: Fiddler organization id
            project: Fiddler project name to publish events to
            model: Fiddler model name to publish events to
            api_key: Fiddler authentication token
        """
        super().__init__()
        # Initialize Fiddler client and other necessary properties
        self.fdl = import_fiddler()
        self.pd = import_pandas()

        self.url = url
        self.org = org
        self.project = project
        self.model = model
        self.api_key = api_key
        self._df = self.pd.DataFrame(_dataset_dict)

        self.run_id_prompts: Dict[UUID, List[str]] = {}
        self.run_id_response: Dict[UUID, List[str]] = {}
        self.run_id_starttime: Dict[UUID, int] = {}

        # Initialize Fiddler client here
        self.fiddler_client = self.fdl.FiddlerApi(url, org_id=org, auth_token=api_key)

        if self.project not in self.fiddler_client.get_project_names():
            print(  # noqa: T201
                f"adding project {self.project}.This only has to be done once."
            )
            try:
                self.fiddler_client.add_project(self.project)
            except Exception as e:
                print(  # noqa: T201
                    f"Error adding project {self.project}:"
                    "{e}. Fiddler integration will not work."
                )
                raise e

        dataset_info = self.fdl.DatasetInfo.from_dataframe(
            self._df, max_inferred_cardinality=0
        )

        # Set feedback column to categorical
        for i in range(len(dataset_info.columns)):
            if dataset_info.columns[i].name == FEEDBACK:
                dataset_info.columns[i].data_type = self.fdl.DataType.CATEGORY
                dataset_info.columns[i].possible_values = FEEDBACK_POSSIBLE_VALUES

            elif dataset_info.columns[i].name == LLM_STATUS:
                dataset_info.columns[i].data_type = self.fdl.DataType.CATEGORY
                dataset_info.columns[i].possible_values = [SUCCESS, FAILURE]

        if self.model not in self.fiddler_client.get_model_names(self.project):
            if self.model not in self.fiddler_client.get_dataset_names(self.project):
                print(  # noqa: T201
                    f"adding dataset {self.model} to project {self.project}."
                    "This only has to be done once."
                )
                try:
                    self.fiddler_client.upload_dataset(
                        project_id=self.project,
                        dataset_id=self.model,
                        dataset={"train": self._df},
                        info=dataset_info,
                    )
                except Exception as e:
                    print(  # noqa: T201
                        f"Error adding dataset {self.model}: {e}."
                        "Fiddler integration will not work."
                    )
                    raise e

            model_info = self.fdl.ModelInfo.from_dataset_info(
                dataset_info=dataset_info,
                dataset_id="train",
                model_task=self.fdl.ModelTask.LLM,
                features=[PROMPT, CONTEXT, RESPONSE],
                target=FEEDBACK,
                metadata_cols=[
                    RUN_ID,
                    TOTAL_TOKENS,
                    PROMPT_TOKENS,
                    COMPLETION_TOKENS,
                    MODEL_NAME,
                    DURATION,
                ],
                custom_features=self.custom_features,
            )
            print(  # noqa: T201
                f"adding model {self.model} to project {self.project}."
                "This only has to be done once."
            )
            try:
                self.fiddler_client.add_model(
                    project_id=self.project,
                    dataset_id=self.model,
                    model_id=self.model,
                    model_info=model_info,
                )
            except Exception as e:
                print(  # noqa: T201
                    f"Error adding model {self.model}: {e}."
                    "Fiddler integration will not work."
                )
                raise e

    @property
    def custom_features(self) -> list:
        """
        Define custom features for the model to automatically enrich the data with.
        Here, we enable the following enrichments:
        - Automatic Embedding generation for prompt and response
        - Text Statistics such as:
            - Automated Readability Index
            - Coleman Liau Index
            - Dale Chall Readability Score
            - Difficult Words
            - Flesch Reading Ease
            - Flesch Kincaid Grade
            - Gunning Fog
            - Linsear Write Formula
        - PII - Personal Identifiable Information
        - Sentiment Analysis

        """

        return [
            self.fdl.Enrichment(
                name="Prompt Embedding",
                enrichment="embedding",
                columns=[PROMPT],
            ),
            self.fdl.TextEmbedding(
                name="Prompt CF",
                source_column=PROMPT,
                column="Prompt Embedding",
            ),
            self.fdl.Enrichment(
                name="Response Embedding",
                enrichment="embedding",
                columns=[RESPONSE],
            ),
            self.fdl.TextEmbedding(
                name="Response CF",
                source_column=RESPONSE,
                column="Response Embedding",
            ),
            self.fdl.Enrichment(
                name="Text Statistics",
                enrichment="textstat",
                columns=[PROMPT, RESPONSE],
                config={
                    "statistics": [
                        "automated_readability_index",
                        "coleman_liau_index",
                        "dale_chall_readability_score",
                        "difficult_words",
                        "flesch_reading_ease",
                        "flesch_kincaid_grade",
                        "gunning_fog",
                        "linsear_write_formula",
                    ]
                },
            ),
            self.fdl.Enrichment(
                name="PII",
                enrichment="pii",
                columns=[PROMPT, RESPONSE],
            ),
            self.fdl.Enrichment(
                name="Sentiment",
                enrichment="sentiment",
                columns=[PROMPT, RESPONSE],
            ),
        ]

    def _publish_events(
        self,
        run_id: UUID,
        prompt_responses: List[str],
        duration: int,
        llm_status: str,
        model_name: Optional[str] = "",
        token_usage_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Publish events to fiddler
        """

        prompt_count = len(self.run_id_prompts[run_id])
        df = self.pd.DataFrame(
            {
                PROMPT: self.run_id_prompts[run_id],
                RESPONSE: prompt_responses,
                RUN_ID: [str(run_id)] * prompt_count,
                DURATION: [duration] * prompt_count,
                LLM_STATUS: [llm_status] * prompt_count,
                MODEL_NAME: [model_name] * prompt_count,
            }
        )

        if token_usage_dict:
            for key, value in token_usage_dict.items():
                df[key] = [value] * prompt_count if isinstance(value, int) else value

        try:
            if df.shape[0] > 1:
                self.fiddler_client.publish_events_batch(self.project, self.model, df)
            else:
                df_dict = df.to_dict(orient="records")
                self.fiddler_client.publish_event(
                    self.project, self.model, event=df_dict[0]
                )
        except Exception as e:
            print(  # noqa: T201
                f"Error publishing events to fiddler: {e}. continuing..."
            )

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        run_id = kwargs[RUN_ID]
        self.run_id_prompts[run_id] = prompts
        self.run_id_starttime[run_id] = int(time.time() * 1000)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        flattened_llmresult = response.flatten()
        run_id = kwargs[RUN_ID]
        run_duration = int(time.time() * 1000) - self.run_id_starttime[run_id]
        model_name = ""
        token_usage_dict = {}

        if isinstance(response.llm_output, dict):
            token_usage_dict = {
                k: v
                for k, v in response.llm_output.items()
                if k in [TOTAL_TOKENS, PROMPT_TOKENS, COMPLETION_TOKENS]
            }
            model_name = response.llm_output.get(MODEL_NAME, "")

        prompt_responses = [
            llmresult.generations[0][0].text for llmresult in flattened_llmresult
        ]

        self._publish_events(
            run_id,
            prompt_responses,
            run_duration,
            SUCCESS,
            model_name,
            token_usage_dict,
        )

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        run_id = kwargs[RUN_ID]
        duration = int(time.time() * 1000) - self.run_id_starttime[run_id]

        self._publish_events(
            run_id, [""] * len(self.run_id_prompts[run_id]), duration, FAILURE
        )

class FiddlerSafetyGuardrailsCallbackHandler(BaseCallbackHandler):
    """Callback Handler that integrates with an external guardrails API.
    
    This handler intercepts LLM responses and sends them to an external
    guardrails API for content moderation. If the response exceeds the
    configured threshold, it will be blocked and an error will be raised.
    """

    def __init__(
        self,
        api_url: str,
        auth_token: str,
        threshold: float = 0.1,
        timeout: int = 5,
        fallback_response: Optional[str] = None,
    ) -> None:
        """Initialize the GuardrailsCallbackHandler.
        
        Args:
            api_url: The URL of the external guardrails API.
            auth_token: The authentication token for the API.
            threshold: The threshold for blocking responses (0.0 to 1.0).
                Defaults to 0.5.
            timeout: Timeout in seconds for API requests. Defaults to 5.
            fallback_response: A fallback response to use when a response is
                blocked and raise_error is False. Defaults to None.
        """
        super().__init__()
        self.api_url = api_url
        self.auth_token = auth_token
        self.threshold = threshold
        self.timeout = timeout
        self.fallback_response = fallback_response
        
        # Store the original responses and their guardrail scores
        self.response_scores: Dict[str, float] = {}
        
        # Store blocked run IDs
        self.blocked_runs: set[UUID] = set()
        
        # Store input scores
        self.input_scores: Dict[str, float] = {}
        
        # Validate configuration
        if not api_url:
            raise ValueError("api_url must be provided")
        if not auth_token:
            raise ValueError("auth_token must be provided")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.api_url += "/v3/guardrails/ftl-safety"

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Check user inputs against the guardrails API before LLM processing.
        
        Args:
            serialized: The serialized LLM.
            prompts: The prompts being sent to the LLM.
            run_id: The ID of the run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """
        # Process each prompt (user input)
        for prompt in prompts:
            # Extract the user input from the prompt
            # This is a simplified approach - you may need to adapt this based on
            # your prompt structure to correctly extract just the user input portion
            user_input = self._extract_user_input(prompt)
            
            if user_input:
                # Check the user input against the guardrails API
                is_blocked, score = self._check_guardrails(user_input)
                
                # Store the score for reference
                self.input_scores[user_input] = score
                
                # If the input is blocked
                if is_blocked:
                    error_msg = (
                        f"User input blocked by guardrails API with score {score} "
                        f"(threshold: {self.threshold})"
                    )
                    logger.warning(error_msg)
                    
                    # Mark this run as blocked so we can replace the response later
                    self.blocked_runs.add(run_id)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Replace responses for blocked inputs with the fallback message.
        
        Args:
            response: The LLM result.
            run_id: The ID of the run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """
        # Check if this run was blocked
        if run_id in self.blocked_runs or (parent_run_id and parent_run_id in self.blocked_runs):
            # Replace all generations with the fallback response
            for generations in response.generations:
                for generation in generations:
                    generation.text = self.fallback_response
                    if hasattr(generation, "message"):
                        generation.message.content = self.fallback_response
            
            # Remove from blocked runs since we've handled it
            if run_id in self.blocked_runs:
                self.blocked_runs.remove(run_id)
            if parent_run_id and parent_run_id in self.blocked_runs:
                self.blocked_runs.remove(parent_run_id)

    def _extract_user_input(self, prompt: str) -> Optional[str]:
        """Extract the user input from a prompt.
        
        This is a simplified implementation. In practice, you'll need to adapt this
        based on your specific prompt structure to correctly extract just the user input.
        
        Args:
            prompt: The full prompt being sent to the LLM.
            
        Returns:
            The extracted user input, or None if it couldn't be extracted.
        """
        # Simple implementation - assumes the last part of the prompt is the user input
        # You'll need to customize this based on your prompt template structure
        
        # Example for a simple chat template where user messages are prefixed with "Human:"
        lines = prompt.split("\n")
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].startswith("Human:"):
                return lines[i][6:].strip()  # Remove "Human:" prefix
        
        # If we can't identify a specific user input, return the whole prompt
        # This is a fallback approach - ideally you'd have a more precise extraction
        return prompt

    def _check_guardrails(self, text: str) -> tuple[bool, float]:
        """Check the text against the guardrails API.
        
        Args:
            text: The text to check.
            
        Returns:
            A tuple of (is_blocked, score).
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "data": {
                    "prompt": [
                        text
                    ]
                }
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract the score (adjust this based on your API's response format)
            score = result[0].get("any_score", 2)

            if score == 2:
                raise ValueError("Invalid score received from guardrails API")
            
            # Determine if the response should be blocked
            is_blocked = score >= self.threshold
            
            return is_blocked, score
            
        except Exception as e:
            logger.error(f"Error checking guardrails API: {e}")
            # Default to not blocking if there's an error with the API
            return False, 0.0

    def get_response_score(self, text: str) -> Optional[float]:
        """Get the guardrail score for a specific response.
        
        Args:
            text: The response text to get the score for.
            
        Returns:
            The score if available, or None if the response hasn't been checked.
        """
        return self.response_scores.get(text)

class FiddlerFaithfulnessGuardrailsCallbackHandler(BaseCallbackHandler):
    """Callback Handler for RAG systems with guardrails integration.
    
    This handler captures both the context from retrievers and the LLM responses
    to provide comprehensive guardrails that can analyze the relationship between
    context and generated content.
    """

    def __init__(
        self,
        api_url: str,
        auth_token: str,
        threshold: float = 0.005,
        timeout: int = 5,
        raise_error: bool = True,
        fallback_response: Optional[str] = None,
    ) -> None:
        """Initialize the RAGGuardrailsCallbackHandler.
        
        Args:
            api_url: The URL of the external guardrails API.
            auth_token: The authentication token for the API.
            threshold: The threshold for blocking responses (0.0 to 1.0).
                Defaults to 0.5.
            timeout: Timeout in seconds for API requests. Defaults to 5.
            raise_error: Whether to raise an error when a response is blocked.
                Defaults to True.
            fallback_response: A fallback response to use when a response is
                blocked and raise_error is False. Defaults to None.
        """
        super().__init__()
        self.api_url = api_url
        self.auth_token = auth_token
        self.threshold = threshold
        self.timeout = timeout
        self.raise_error = raise_error
        self.fallback_response = fallback_response
        
        # Store the retrieved documents for each run
        self.retrieved_documents: Dict[UUID, List[str]] = {}
        
        # Store the prompts for each run
        self.prompts: Dict[UUID, str] = {}
        
        # Store the original responses and their guardrail scores
        self.response_scores: Dict[str, float] = {}
        
        # Validate configuration
        if not api_url:
            raise ValueError("api_url must be provided")
        if not auth_token:
            raise ValueError("auth_token must be provided")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")

        self.api_url += "/v3/guardrails/ftl-response-faithfulness"

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Capture the retrieved documents.
        
        Args:
            documents: The retrieved documents.
            run_id: The ID of the run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """
        # Store the document contents for this run
        self.retrieved_documents[run_id] = [doc.page_content for doc in documents]
        
        # If there's a parent run, also store under the parent run ID
        if parent_run_id:
            if parent_run_id not in self.retrieved_documents:
                self.retrieved_documents[parent_run_id] = []
            self.retrieved_documents[parent_run_id].extend([doc.page_content for doc in documents])

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Check the LLM response against the guardrails API.
        
        Args:
            response: The LLM result to check.
            run_id: The ID of the run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
            
        Raises:
            ValueError: If the response is blocked by the guardrails API and
                raise_error is True.
        """
        # Get the context for this run
        context = self.retrieved_documents.get(run_id, [])
        if not context and parent_run_id:
            context = self.retrieved_documents.get(parent_run_id, [])
        
        # Process each generation in the response
        for generations in response.generations:
            for generation in generations:
                # Check the text against the guardrails API
                is_blocked, score = self._check_guardrails(
                    response_text=generation.text,
                    context=context,
                )
                
                # Store the score for reference
                self.response_scores[generation.text] = score
                
                # If the response is blocked and we're configured to raise an error
                if is_blocked and self.raise_error:
                    error_msg = (
                        f"Response blocked by guardrails API with score {score} "
                        f"(threshold: {self.threshold})"
                    )
                    logger.warning(error_msg)
                    raise ValueError(error_msg)
                
                # If the response is blocked and we have a fallback
                if is_blocked and self.fallback_response is not None:
                    # Modify the generation in place
                    generation.text = self.fallback_response
                    if hasattr(generation, "message"):
                        generation.message.content = self.fallback_response

    def _check_guardrails(
        self, 
        response_text: str, 
        context: List[str],
    ) -> tuple[bool, float]:
        """Check the response against the guardrails API, including context.
        
        Args:
            response_text: The LLM response text to check.
            context: The retrieved document contents.
            prompt: The prompt sent to the LLM.
            
        Returns:
            A tuple of (is_blocked, score).
        """
        try:
            headers = {
                "Authorization": f"Bearer {self.auth_token}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "response": response_text,
                "context": context,
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            
            # Check if the request was successful
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Extract the score (adjust this based on your API's response format)
            score = result.get("score", 0.0)
            
            # Determine if the response should be blocked
            is_blocked = score >= self.threshold
            
            return is_blocked, score
            
        except Exception as e:
            logger.error(f"Error checking guardrails API: {e}")
            # Default to not blocking if there's an error with the API
            return False, 0.0

    def get_response_score(self, text: str) -> Optional[float]:
        """Get the guardrail score for a specific response.
        
        Args:
            text: The response text to get the score for.
            
        Returns:
            The score if available, or None if the response hasn't been checked.
        """
        return self.response_scores.get(text)