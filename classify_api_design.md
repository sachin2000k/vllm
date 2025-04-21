# Classify API Design for vllm

## Overview

The goal is to add a new API endpoint to vllm that returns classification head logits, which users can then use with softmax or sigmoid to get probabilities. This extends the current functionality where the pooling API returns the last hidden layer logits.

## Current Implementation Analysis

From analyzing the vllm codebase, I've found:

1. The `LLM` class already has a `classify` method that returns class probabilities
2. The pooling API is implemented in `vllm/entrypoints/openai/serving_pooling.py`
3. The model execution for pooling is handled by `vllm/worker/pooling_model_runner.py`
4. The pooling implementation is in `vllm/model_executor/layers/pooler.py`
5. There's an example of classification usage in `examples/offline_inference/basic/classify.py`

## Design Approach

To implement the classify API, we'll follow these design principles:

1. **Extend the OpenAI-compatible API**: Create a new endpoint similar to the pooling endpoint
2. **Reuse existing code**: Leverage the existing classification functionality in the LLM class
3. **Maintain consistency**: Follow the same patterns as the pooling API for request/response handling
4. **Provide flexibility**: Allow users to get raw logits that they can process with softmax/sigmoid

## API Design

### 1. Protocol Classes

We'll need to add new protocol classes in `vllm/entrypoints/openai/protocol.py`:

```python
class ClassifyRequest(PoolingRequest):
    """Request for the classify endpoint."""
    # Inherits from PoolingRequest since most parameters are the same
    # We might add additional parameters specific to classification if needed

class ClassifyResponseData(BaseModel):
    """Data for a single item in a classify response."""
    index: int
    logits: Union[list[float], str]  # Either raw logits or base64 encoded

class ClassifyResponse(BaseModel):
    """Response for the classify endpoint."""
    id: str
    created: int
    model: str
    data: list[ClassifyResponseData]
    usage: UsageInfo
```

### 2. Serving Class

We'll create a new class `OpenAIServingClassify` in a new file `vllm/entrypoints/openai/serving_classify.py` that extends `OpenAIServing`:

```python
class OpenAIServingClassify(OpenAIServing):
    """Serving class for the classify endpoint."""
    
    def __init__(self, ...):
        # Similar to OpenAIServingPooling initialization
        
    async def create_classify(self, request: ClassifyRequest, raw_request: Optional[Request] = None) -> Union[ClassifyResponse, ErrorResponse]:
        """Handle classify requests and return classification head logits."""
        # Similar to create_pooling but returns classification head logits
```

### 3. API Endpoint

We'll add a new endpoint to `vllm/entrypoints/openai/api_server.py`:

```python
@app.post("/v1/classify")
async def create_classify(
    request: ClassifyRequest,
    raw_request: Request = Depends(get_raw_request),
) -> Union[ClassifyResponse, ErrorResponse]:
    """Create classification head logits for the provided input."""
    return await serving_classify.create_classify(request, raw_request)
```

### 4. Model Runner Modifications

We'll need to modify the `PoolingModelRunner` class or create a new `ClassifyModelRunner` class to handle classification requests and return the classification head logits instead of the last hidden layer logits.

### 5. Integration with Existing Code

We'll integrate with the existing classification functionality in the LLM class, ensuring that the server API uses the same underlying implementation.

## Implementation Plan

1. Add new protocol classes in `protocol.py`
2. Create the `serving_classify.py` file with the `OpenAIServingClassify` class
3. Add the new endpoint to `api_server.py`
4. Modify or extend the model runner to handle classification requests
5. Update the OpenAI API server initialization to include the classify endpoint

## Testing Strategy

1. Unit tests for the new protocol classes
2. Integration tests for the classify endpoint
3. End-to-end tests comparing results with the existing LLM.classify method

## Documentation Updates

1. Update API documentation to include the new classify endpoint
2. Add examples of how to use the classify endpoint
3. Explain the difference between pooling and classify endpoints
