from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from backend.core import settings


def _resolve_model(base_url: str, project: str, model: str) -> str:
    if project and "cloud.yandex.net" in base_url and not model.startswith("gpt://"):
        return f"gpt://{project}/{model}"
    return model


async def main() -> None:
    print(f"{os.getenv('LLM_API_KEY')=}")
    print(f"{settings=}")
    client = AsyncOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        project=settings.llm_project or None,
        timeout=settings.llm_timeout_s,
    )
    model = _resolve_model(
        base_url=settings.llm_base_url,
        project=settings.llm_project,
        model=settings.llm_model,
    )

    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": "You evaluate Wikipedia documents and decide whether they are enough to answer the question.\nReturn JSON only with keys: enough_information, findings, missing_information, next_wikipedia_queries.\nRules:\n- enough_information: true only if facts are sufficient for a grounded answer.\n- findings: 1-8 concise factual bullets extracted from provided documents.\n- findings must be grounded strictly in provided excerpts.\n- missing_information: short bullet-like strings describing what is still missing.\n- next_wikipedia_queries must contain plain entity titles only.\n- Return empty next_wikipedia_queries if enough_information is true.\n- Avoid repeating already executed queries unless absolutely necessary."}, {"role": "user", "content": "Question:\nНасколько трамп старше путина\n\nPlan:\n['Collect birth dates for Donald Trump and Vladimir Putin from reliable sources like Wikipedia.', \"Calculate current ages of both individuals based on today's date.\", 'Compute the age difference in years, months, and days.', 'Verify the birth dates and calculations with additional sources for accuracy.', 'Perform a consistency self-check to ensure no errors in the computation.']\n\nAssumptions:\n['Donald Trump and Vladimir Putin have publicly known birth dates.', 'Their ages can be calculated accurately based on current date.', 'The age difference is typically expressed in years.']\n\nExecuted wikipedia queries:\n['donald trump', 'vladimir putin']\n\nPrevious findings:\n[]\n\nWikipedia documents (trimmed excerpts):\n[1] query=Donald Trump; title=Donald Trump; truncated=True; excerpt=\"Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States. A member of the Republican Party, he served as the 45th president from 2017 to 2021.\\nBorn into a wealthy New York City family, Trump graduated from the University of Pennsylvania in 1968 with a bachelor's degree in economics. He became the president of his family's real estate business in 1971, renamed it the Trump Organization, and began acquiring and building skyscrapers, hotels, casinos, and golf courses. He launched side ventures, many licensing the Trump name, and filed for six business bankruptcies in the \"\n[2] query=Vladimir Putin; title=Vladimir Putin; truncated=True; excerpt=\"Vladimir Vladimirovich Putin (born 7 October 1952) is a Russian politician and former intelligence officer who has served as President of Russia since 2012, having previously served from 2000 to 2008. Putin also served as Prime Minister of Russia from 1999 to 2000 and again from 2008 to 2012. He has been described as the de facto leader of Russia since 1999.\\nBorn in Leningrad (now Saint Petersburg), Putin worked as a KGB fo...<truncated 259 chars>"}],
    )
    content = response.choices[0].message.content
    print(content if isinstance(content, str) else repr(content))


if __name__ == "__main__":
    asyncio.run(main())
