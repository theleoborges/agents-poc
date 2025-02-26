from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools  import SerperDevTool

llm = LLM(model="gpt-4o", temperature=0.5)


researcher = Agent(
    role="Senior Research Analyst",
    goal="Find comprehensive and accurate information on the assigned topic",
    backstory="""You are an experienced research analyst with a talent for
    finding reliable and relevant information. You're thorough and always verify
    your sources.""",
    verbose=True,
    allow_delegation=False,
    tools=[SerperDevTool()],
    llm_model=llm

)

analyst = Agent(
    role="Data Analyst",
    goal="Analyze research data to identify key insights and trends",
    backstory="""You excel at synthesizing information and extracting
    valuable insights. You can spot patterns that others miss and are skilled
    at organizing complex information.""",
    verbose=True,
    allow_delegation=False,
    llm_model=llm
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging and informative content based on research and analysis",
    backstory="""You are a talented writer who can transform complex
    information into clear, engaging content. You adapt your writing style to
    match the intended audience and purpose.""",
    verbose=True,
    allow_delegation=False,
    llm_model=llm
)

editor = Agent(
    role="Content Editor",
    goal="Ensure content is polished, accurate, and meets quality standards",
    backstory="""You have a keen eye for detail and are skilled at
    refining content to make it more effective. You check for consistency,
    clarity, and accuracy.""",
    verbose=True,
    allow_delegation=False,
    llm_model=llm
)

# Define the topic for this run (you can change this to any topic)
topic = "The impact of artificial intelligence on software development. Specifically regarding tools like Github CoPilot, Cursor and the recently released Claude Code."

# Define tasks
research_task = Task(
    description=f"""Research the following topic thoroughly: {topic}. 
    Focus on recent developments, key applications, challenges, and future trends. 
    Provide comprehensive research notes and data with citations where possible.""",
    agent=researcher,
    expected_output=f"Detailed research notes on {topic} with key findings and sources"
)

analysis_task = Task(
    description="""Review the research notes and analyze the information to identify key insights, patterns, and implications. 
    Organize the information into a structured analytical framework.
    Include citations where possible.""",
    agent=analyst,
    expected_output="Analytical report with key insights and structured framework",
    context=[research_task]
)

writing_task = Task(
    description="""Using the analytical report, create an engaging and informative article suitable for a general audience. 
    The article should be well-structured with an introduction, main sections covering key points, and a conclusion.
    Be sure to include data and citations to substantiate claims where possible.""",
    agent=writer,
    expected_output="Well-written article draft based on the research, analysis and data provided.",
    context=[analysis_task]
)

editing_task = Task(
    description="""Review and edit the article draft to ensure it is polished, accurate, engaging, and effectively communicates the key information. 
    Check for clarity, flow, grammar, and overall quality. Include references at the end of the article.""",
    agent=editor,
    expected_output="Final polished article ready for publication",
    context=[writing_task]
)


def main():
    # Create the crew
    content_crew = Crew(
        agents=[researcher, analyst, writer, editor],
        tasks=[research_task, analysis_task, writing_task, editing_task],
        verbose=True,
        process=Process.sequential  # Tasks will be executed in sequence
    )

    # Run the crew and get the result
    result = content_crew.kickoff()

    print("\n\n========================")
    print("FINAL OUTPUT:")
    print(result)

if __name__ == "__main__":
    main()