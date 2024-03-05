from dotenv import load_dotenv
import os
import pandas as pd
from llama_index.core.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.agent import ReActAgent
from llama_index.llms import OpenAI
from pdf import canada_engine



# Load population data from a CSV file
population_path = os.path.join("data", "population.csv")
population_df = pd.read_csv(population_path)

# Initialize the PandasQueryEngine for population data
population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
)
population_query_engine.update_prompts({"pandas_prompt": new_prompt})

# Define tools
tools = [
    note_engine,
    QueryEngineTool(
        query_engine=population_query_engine,
        metadata=ToolMetadata(
            name="population_data",
            description="This provides information on world population and demographics.",
        ),
    )
]

# Initialize the GPT-3.5-Turbo model
llm = OpenAI(model="gpt-3.5-turbo-0613")

# Create an agent with the specified tools and language model
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)

# User interaction loop
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)
