# SPDX-License-Identifier: Apache-2.0

from typing import Literal, Optional, Union

from fastapi import Request
from pydantic import BaseModel, Field

from vllm.entrypoints.openai.protocol import (ErrorResponse, OpenAIBaseModel,
                                             PoolingCompletionRequest, PoolingChatRequest, UsageInfo)


# Create separate classes for completion and chat requests
class ClassifyCompletionRequest(PoolingCompletionRequest):
    """Request for the classify endpoint using completion format."""
    classification_type: Literal["multiclass", "multilabel"]

    pass


class ClassifyChatRequest(PoolingChatRequest):
    """Request for the classify endpoint using chat format."""
    classification_type: Literal["multiclass", "multilabel"]

    pass


# Define the union type for the API endpoint
ClassifyRequest = Union[ClassifyCompletionRequest, ClassifyChatRequest]


class ClassifyResponseData(BaseModel):
    """Data for a single item in a classify response."""
    index: int
    logits: Union[list[float], str]  # Either raw logits or base64 encoded
    probabilities: list[float] = []


class ClassifyResponse(OpenAIBaseModel):
    """Response for the classify endpoint."""
    id: str
    created: int
    model: str
    data: list[ClassifyResponseData]
    usage: UsageInfo
