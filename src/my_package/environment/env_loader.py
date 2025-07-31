import getpass
import os


def load_env() -> None:
    """環境変数を読み込む"""
    try:
        # .envファイルから環境変数を読み込む
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    os.environ["LANGSMITH_TRACING"] = "true"
    if "LANGSMITH_API_KEY" not in os.environ:
        os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
            prompt="Enter your LangSmith API key (optional): "
        )
    if "LANGSMITH_PROJECT" not in os.environ:
        os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
            prompt='Enter your LangSmith Project Name (default = "default"): '
        )
        if not os.environ.get("LANGSMITH_PROJECT"):
            os.environ["LANGSMITH_PROJECT"] = "default"

    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

    # Set USER_AGENT for web scraping requests
    if not os.environ.get("USER_AGENT"):
        os.environ["USER_AGENT"] = "LangChain-RAG-Bot/1.0 (Educational Project)"
