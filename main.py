from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools  import SerperDevTool

llm = LLM(model="claude-3-7-sonnet-20250219", temperature=0.5)


researcher = Agent(
    role="Senior Research Analyst",
    goal="Conduct exhaustive research on the assigned topic, gathering comprehensive data from multiple sources",
    backstory="""You are a world-class research analyst with a talent for finding detailed, reliable information
    from diverse sources. You're meticulous, thorough, and always verify your sources.
    You specialize in deep research that uncovers insights others might miss.""",
    verbose=True,
    allow_delegation=False,
    tools=[SerperDevTool(n_results=50)],
    llm_model=llm

)

analyst = Agent(
    role="Data Analyst",
    goal="Perform in-depth analysis of research data to identify significant insights, trends, and implications",
    backstory="""You are an expert analyst with extensive domain knowledge. You excel at synthesizing complex information
    and extracting valuable insights. You can connect dots across disparate data points and identify both
    obvious and subtle patterns. You pride yourself on transforming raw research into structured, actionable knowledge.""",
    verbose=True,
    allow_delegation=False,
    llm_model=llm
)

writer = Agent(
    role="Content Writer",
    goal="Create comprehensive, engaging, and authoritative content that thoroughly explores the subject matter",
    backstory="""You are an exceptional writer with expertise in creating long-form, in-depth content.
    You know how to structure complex information into readable, engaging narratives that maintain reader interest
    despite the depth and detail. You excel at explaining complex concepts clearly and compellingly,
    and you always ensure your writing is substantiated with evidence.""",
    verbose=True,
    allow_delegation=False,
    llm_model=llm
)

editor = Agent(
    role="Content Editor",
    goal="Ensure the final content is comprehensive, accurate, cohesive, and meets the highest standards of quality",
    backstory="""You are a senior editor with decades of experience in polishing and perfecting substantial content.
    You have an eagle eye for detail and a talent for improving both the micro elements (grammar, style, flow)
    and macro elements (structure, argumentation, completeness) of content. You ensure every piece meets the
    highest standards of quality and comprehensiveness.""",
    verbose=True,
    allow_delegation=False,
    llm_model=llm
)

# Define the topic for this run (you can change this to any topic)
topic = "The risks of not adopting proactive vulnerability scanning tools like Snyk. Include aspects regarding AI assisted remediation."

# Define tasks
research_task = Task(
    description=f"""Conduct extensive research on the following topic: {topic}.
    
    Your research must be comprehensive and in-depth. Search for and compile information from:
    - Recent academic publications and peer-reviewed journals
    - Industry reports and white papers
    - Expert opinions and interviews
    - Case studies and real-world applications
    - Statistical data and trends
    - Multiple perspectives including enthusiasts and critics
    
    For each subtopic or angle:
    1. Find at least 3-5 distinct authoritative sources and cite them
    2. Gather specific examples, statistics, quotes, and detailed information
    3. Note contradictory views or debates within the field
    4. Identify emerging trends and future predictions
    
    Organize your research into detailed sections covering:
    - Current state and major developments (with emphasis on 2025)
    - Technical aspects and implementation details
    - Real-world applications and case studies
    - Challenges and limitations
    - Ethical considerations and debates
    - Future prospects and trends
    
    IMPORTANT: When searching for information, always use 2024 and 2025 as years and focus on the most recent information available.
    DO NOT add older years to your search queries. Always specify the years 2024 and 2025 in your searches to get the most up-to-date information.
    
    Your final research document should be extensive, containing 5,000+ words of raw information,
    properly organized with headings and subheadings, and including full citations and links for all sources.
    This will serve as the foundation for an authoritative, in-depth report.
    
    If you need clarification or specialized information, delegate specific research questions to other agents.
    """,
    agent=researcher,
    expected_output="Exhaustive research document with comprehensive information, multiple perspectives, and detailed citations and links"
)

analysis_task = Task(
    description="""Perform an in-depth analysis of the research document to transform raw information into structured, valuable insights.
    
    Your analysis should:
    1. Identify major themes, patterns, and trends across the research
    2. Evaluate the significance, reliability, and implications of different information
    3. Connect related concepts and identify cause-effect relationships
    4. Highlight contradictions, knowledge gaps, or areas of uncertainty
    5. Compare and contrast different approaches, methodologies, or viewpoints
    6. Develop frameworks to organize and understand the information
    7. Draw evidence-based conclusions about current status and future directions
    
    Structure your analysis into these components:
    - Executive summary of key findings (500+ words)
    - Detailed analysis organized by major themes (3000+ words)
    - Evaluation of evidence quality and consensus levels
    - Identification of emerging patterns and their implications
    - Critical assessment of challenges and opportunities
    - Synthesis of interdisciplinary connections
    - Detailed future outlook with short and long-term projections
    
    Your analytical document should be at least 5,000 words and include visual frameworks (described in text)
    that help conceptualize the information. Maintain a balanced, objective perspective while
    providing insightful interpretation of the data.
    
    If you need additional research or clarification on specific points, delegate these needs to the researcher.
    """,
    agent=analyst,
    expected_output="Comprehensive analytical report with structured insights, frameworks, and evidence-based conclusions that include citations and references with links.",
    context=[research_task]
)

writing_task = Task(
    description="""Create an authoritative, in-depth article based on the research and analysis provided.
    
    Your article should be comprehensive, engaging, and professionally structured. Aim for approximately 5,000-8,000 words
    of substantial content that thoroughly explores the topic and provides valuable insights to readers.
    
    The article should include:
    
    1. An engaging introduction that establishes context, significance, and scope (500+ words)
    
    2. A well-organized body with clear sections and subsections:
       - Develop each major point with sufficient depth (500+ words per major section)
       - Include concrete examples, case studies, and specific details
       - Present multiple perspectives and balanced viewpoints
       - Explain complex concepts thoroughly but accessibly
       - Address counterarguments and limitations
       - Use transitional elements to maintain flow between sections
    
    3. Rich supporting elements (described in text form):
       - Statistical data with interpretation
       - Expert viewpoints with direct quotes where available
       - Detailed examples and case studies
       - Frameworks for understanding complex concepts
    
    4. A substantial conclusion that synthesizes key insights and looks forward (500+ words)
    
    5. Consider adding these additional elements to enhance depth:
       - Technical deep-dive sections for knowledgeable readers
       - "Future implications" section with evidence-based predictions
       - "Practical applications" section with real-world examples
       - "Limitations and challenges" section with nuanced discussion
    
    Write in a professional, authoritative tone appropriate for an educated audience while keeping the content
    accessible and engaging. Balance breadth with depth, ensuring comprehensive coverage while maintaining
    reader interest through compelling writing and varied structure.
    
    If you need additional analysis or research to properly develop any section, delegate these requests to the appropriate agent.
    """,
    agent=writer,
    expected_output="Authoritative, comprehensive article (5,000-8,000 words) with in-depth exploration of all aspects of the topic, including citations and references with links.",
    context=[analysis_task]
)

editing_task = Task(
    description="""Thoroughly review and enhance the comprehensive article to ensure it meets the highest standards of quality,
    depth, and authority.
    
    Your editorial review should be extensive and meticulous, focusing on both content substance and presentation quality.
    
    Content Enhancement:
    1. Verify the article comprehensively covers all important aspects of the topic
    2. Ensure proper depth for each section (at least 500 words per major section)
    3. Confirm the presence of specific examples, data, and evidence throughout
    4. Check that multiple perspectives are presented in a balanced manner
    5. Identify and fill any remaining content gaps or shallow areas
    6. Strengthen the argumentation and logical flow throughout
    7. Ensure conclusions are fully supported by the presented evidence
    
    Structural Improvement:
    1. Optimize the overall structure for logical progression and reader understanding
    2. Ensure appropriate balance between sections
    3. Verify that headings and subheadings effectively organize the content
    4. Add transitional elements where needed to improve flow
    
    Quality Assurance:
    1. Correct any grammatical, spelling, or punctuation errors
    2. Refine language for clarity, precision, and engagement
    3. Eliminate redundancies and tighten prose while maintaining depth
    4. Check citation format and completeness
    5. Ensure consistent tone and style throughout
    
    Final Enhancements:
    1. Strengthen the introduction to effectively hook readers
    2. Enhance the conclusion to leave a lasting impression
    3. Add clarifying elements where complex concepts are presented
    4. Ensure the article maintains reader interest despite its length and depth
    5. Ensure the article has appropriate citations and references with links
    
    Your final edit should result in a polished, authoritative piece of 5,000-8,000 words that stands as 
    a definitive resource on the topic. The article should be publication-ready and meet the standards
    of respected industry publications or academic journals.
    
    If you identify areas that require additional research or content development, delegate these 
    specific needs to the appropriate agent.
    """,
    agent=editor,
    expected_output="Publication-ready, comprehensive article (5,000-8,000 words) that serves as an authoritative resource on the topic",
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
    print("FINAL OUTPUT: output.md")
    print(result)

    with open('output.md', 'w') as file:
        file.write(result.raw)

    

if __name__ == "__main__":
    main()