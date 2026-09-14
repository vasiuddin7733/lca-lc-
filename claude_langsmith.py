from dotenv import load_dotenv
from langsmith import traceable
from anthropic import Anthropic

load_dotenv()

client = Anthropic()


@traceable
def format_prompt(subject):
    return f"Tell me about {subject}"


@traceable(run_type="llm")
def invoke_llm(prompt):
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
    )

    return response


@traceable
def parse_output(response):
    return response.content[0].text


@traceable
def run_pipeline():
    prompt = format_prompt("foo")
    response = invoke_llm(prompt)
    return parse_output(response)


if __name__ == "__main__":
    result = run_pipeline()
    print(result)