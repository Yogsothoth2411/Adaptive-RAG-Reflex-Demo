from dotenv import load_dotenv
from src.workflow.graph import app, memory
from langchain_core.messages import HumanMessage

load_dotenv()
config = {"configurable": {"thread_id": "abc123"}}


def format_response(result):
    """Extract response from workflow result."""
    if isinstance(result, dict) and "generation" in result:
        return result["generation"]
    elif isinstance(result, dict) and "answer" in result:
        return result["answer"]
    else:
        return str(result)


def main():
    """CLI for adaptive RAG system."""
    print("Adaptive RAG System")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in ["quit", "exit", "q", ""]:
                break
            answer = app.invoke(  # 因為有做循環（如反思→重新生成）、router直接導向生成節點，所以要傳入空值處理初始狀態，否則會得到KeyError
                {
                    "question": question,
                    "chat_history": HumanMessage(content=question),
                    "documents": [],
                    "generation": [],
                    "loop_count": 1,
                    "prev_node": "",
                    "reflex_prompt": "",
                    "sub_questions": [],
                    "web_search": False,
                },
                config,
            )
            print("Processing...")
            result = None
            for output in answer:
                result = answer
            if result:
                print(f"\nAnswer: {format_response(result)}")
            else:
                print("No response generated.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
