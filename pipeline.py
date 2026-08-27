from agent import build_search_agent , build_reader_agent , writer_chain, critic_chain

def run_search_pipeline(topic:str)->dict:
    state = {}
    #search agent working
    print("\n"+"="*50)
    print("1 step - search agent is working")
    print("\n"+"="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user",f"Find reacent realiable and detailed information about: {topic}")]
    })
    state["search_result"] = search_result['messages'][-1].content
    print("search_result : ",state['search_result'])

    #step 2 render agent
    print("\n"+"="*50)
    print("2 step - Reader agent is scraping top resources")
    print("\n"+"="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages":[("user",
                    f"based on the following search reasult about '{topic}'"
                    f"pick the most realivant URL and scrape it dor deaper content.\n\n"
                    f"Search Reasult:\n{state['search_result'][:800]}"
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content

    # Step 3: Writer
    print("\n" + "=" * 50)
    print("3 step - Writer is drafting the report")
    print("=" * 50)

    research_combined = (
        f"SEARCH RESULT:\n{state['search_result']}\n\n"
        f"DETAILED SCRAPED RESULT:\n{state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\nFinal Report:\n")
    print(state["report"])

    # Step 4: Critic
    print("\n" + "=" * 50)
    print("4 step - Critic is reviewing the report")
    print("=" * 50)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\nCritic Report:\n")
    print(state["feedback"])

    return state

if __name__ == "__main__":
    topic = input("\n Enter the topic for research : ")
    run_search_pipeline(topic)