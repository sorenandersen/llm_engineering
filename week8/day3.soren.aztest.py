import os
from typing import Optional, List
from openai import AzureOpenAI
from agents.deals import DealSelection

from dotenv import load_dotenv
load_dotenv()
MODEL = os.getenv("AZURE_OPENAI_MODEL_DEPLOYMENT_NAME") # gpt-4o-mini
openai = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)


class ScrapedDeal:
    """
    A class to represent a Deal retrieved from an RSS feed
    """
    def __init__(self, category: str, title: str, summary: str, url: str, details: str, features: str):
        self.category = category
        self.title = title
        self.summary = summary
        self.url = url
        self.details = details
        self.features = features

    def __repr__(self):
        """
        Return a string to describe this deal
        """
        return f"<{self.title}>"

    def describe(self):
        """
        Return a longer string to describe this deal for use in calling a model
        """
        return f"Title: {self.title}\nDetails: {self.details.strip()}\nFeatures: {self.features.strip()}\nURL: {self.url}"



# Create a list of 3 test ScrapedDeal objects
test_deals = [
    ScrapedDeal(
        category="Electronics",
        title="50% Off Premium Headphones",
        summary="Top-brand noise-cancelling headphones at half price",
        url="https://example.com/deal/headphones",
        details="Limited-time offer. High-fidelity sound and comfortable over-ear design. Price only $69.99",
        features="Wireless, Active Noise Cancellation, 30-hour battery life"
    ),
    # ScrapedDeal(
    #     category="Kitchen",
    #     title="Discounted Espresso Machine",
    #     summary="Up to 40% off on select espresso machines",
    #     url="https://example.com/deal/espresso-machine",
    #     details="Brew café-quality espresso at home with ease. Prices starting at $179.99",
    #     features="Built-in grinder, Milk frother, Adjustable brew strength"
    # ),
    # ScrapedDeal(
    #     category="Travel",
    #     title="Budget Flights to Europe",
    #     summary="Fly to popular European destinations at reduced rates",
    #     url="https://example.com/deal/flights-europe",
    #     details="Book now for discounted round-trip tickets across major EU cities.",
    #     features="Flexible travel dates, Multiple departure cities, No booking fees"
    # )
]

SYSTEM_PROMPT = """You identify and summarize the 5 most detailed deals from a list, by selecting deals that have the most detailed, high quality description and the most clear price.
Respond strictly in JSON with no explanation, using this format. You should provide the price as a number derived from the description. If the price of a deal isn't clear, do not include that deal in your response.
Most important is that you respond with the 5 deals that have the most detailed product description with price. It's not important to mention the terms of the deal; most important is a thorough description of the product.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 

{"deals": [
    {
        "product_description": "Your clearly expressed summary of the product in 4-5 sentences. Details of the item are much more important than why it's a good deal. Avoid mentioning discounts and coupons; focus on the item itself. There should be a paragpraph of text for each item you choose.",
        "price": 99.99,
        "url": "the url as provided"
    },
    ...
]}"""

USER_PROMPT_PREFIX = """Respond with the most promising 5 deals from this list, selecting those which have the most detailed, high quality product description and a clear price that is greater than 0.
Respond strictly in JSON, and only JSON. You should rephrase the description to be a summary of the product itself, not the terms of the deal.
Remember to respond with a paragraph of text in the product_description field for each of the 5 items that you select.
Be careful with products that are described as "$XXX off" or "reduced by $XXX" - this isn't the actual price of the product. Only respond with products when you are highly confident about the price. 

Deals:

"""

USER_PROMPT_SUFFIX = "\n\nStrictly respond in JSON and include exactly 5 deals, no more."


def make_user_prompt(scraped) -> str:
    """
    Create a user prompt for OpenAI based on the scraped deals provided
    """
    user_prompt = USER_PROMPT_PREFIX
    user_prompt += '\n\n'.join([scrape.describe() for scrape in scraped])
    user_prompt += USER_PROMPT_SUFFIX
    return user_prompt

def scan() -> Optional[DealSelection]:
    """
    Call OpenAI to provide a high potential list of deals with good descriptions and prices
    Use StructuredOutputs to ensure it conforms to our specifications
    :param memory: a list of URLs representing deals already raised
    :return: a selection of good deals, or None if there aren't any
    """
    print("scan()")
    scraped = test_deals
    if scraped:
        user_prompt = make_user_prompt(scraped)
        print("Scanner Agent is calling OpenAI using Structured Output")
        print(f"1 MODEL: {MODEL}")
        print(f"2 SYSTEM_PROMPT: {SYSTEM_PROMPT}")
        print(f"3 USER_PROMPT: {user_prompt}")
        print(f"-------------")
        result = openai.beta.chat.completions.parse(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            response_format=DealSelection
        )
        result = result.choices[0].message.parsed
        result.deals = [deal for deal in result.deals if deal.price>0]
        return result
    return None


result = scan()
print(result)
