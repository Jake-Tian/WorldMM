from typing import Dict, List, Union, Any, Optional
from worldmm.llm import generate_text_response
from .logging_utils import get_logger

logger = get_logger(__name__)

def merge_elements_with_same_first_line(elements, prefix='Wikipedia Title: '):
...
def reason_step(dataset, prompt_template_manager, query: str, passages: list, thoughts: list, model_name: str = "gpt-5-mini"):
    """
    Given few-shot samples, query, previous retrieved passages, and previous thoughts, generate the next thought with OpenAI models. The generated thought is used for further retrieval step.
    :return: next thought
    """

    prompt_user = ''
    if dataset in ['hotpotqa', 'hotpotqa_train']:
        passages = merge_elements_with_same_first_line(passages)
    for passage in passages:
        prompt_user += f'{passage}\n\n'
    prompt_user += f'Question: {query}\nThought:' + ' '.join(thoughts)


    messages = prompt_template_manager.render(name=f'ircot_{dataset}', prompt_user=prompt_user)

    try:
        response_content, tokens = generate_text_response(messages=messages, model=model_name)
    except Exception as e:
        logger.exception("An exception occurred while calling LLM for QA!")
        return ''

    return response_content