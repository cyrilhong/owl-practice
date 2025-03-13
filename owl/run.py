from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.toolkits import (
    AudioAnalysisToolkit,
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    VideoAnalysisToolkit,
    BrowserToolkit,
    FileWriteToolkit,
)
from camel.types import ModelPlatformType, ModelType
from camel.logger import set_log_level
from utils import OwlRolePlaying, run_society, DocumentProcessingToolkit
import time
import os

load_dotenv()
set_log_level(level="DEBUG")

print(f"GOOGLE_API_KEY: {os.getenv('GOOGLE_API_KEY')}")
print(f"SEARCH_ENGINE_ID: {os.getenv('SEARCH_ENGINE_ID')}")


def construct_society(question: str) -> OwlRolePlaying:
    models = {
        "user": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0, "max_tokens": 4000},
        ),
        "assistant": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0, "max_tokens": 4000},
        ),
        "web": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0, "max_tokens": 4000},
        ),
        "planning": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0, "max_tokens": 4000},
        ),
        "video": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0, "max_tokens": 4000},
        ),
        "image": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0, "max_tokens": 4000},
        ),
        "document": ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=ModelType.GPT_4O_MINI,
            model_config_dict={"temperature": 0, "max_tokens": 4000},
        ),
    }
    tools = [
        *BrowserToolkit(
            headless=False,
            web_agent_model=models["web"],
            planning_agent_model=models["planning"],
        ).get_tools(),
        *VideoAnalysisToolkit(model=models["video"]).get_tools(),
        *AudioAnalysisToolkit().get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_google,
        SearchToolkit().search_wiki,
        *ExcelToolkit().get_tools(),
        *DocumentProcessingToolkit(model=models["document"]).get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),
    ]
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}
    task_kwargs = {"task_prompt": question, "with_task_specify": False}
    return OwlRolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )


def main():
    r"""Main function to run the OWL system with an example question."""
    question = "搜集台灣的酒類相關課程或證照資訊，並整理成表格，每個證照為單位，表格需包含以下欄位：課名、課程資訊、時間、地點、費用。步驟：1) 優先使用 Google 搜尋 '台灣 酒類 課程 證照'（使用 GOOGLE_API_KEY 和 SEARCH_ENGINE_ID，num_result_pages=3），若失敗則使用 DuckDuckGo 搜尋相同關鍵詞；2) 從結果中選取 3-5 個證照相關課程；3) 針對每個證照，從搜尋結果或相關網站提取課名、課程資訊（簡述課程內容）、時間（開課日期或時段）、地點（具體地點或城市）、費用（具體金額，若無則標記為需進一步確認），並將提取的資訊整理成表格；4) 將表格儲存為 CSV 檔案 'alcohol_courses.csv'，使用 FileWriteToolkit 的 write_file 工具，檔案路徑為 './alcohol_courses.csv'，確保 CSV 包含表頭（課名,課程資訊,時間,地點,費用）和資料列。"
    society = construct_society(question)
    max_retries = 3
    for attempt in range(max_retries):
        try:
            answer, chat_history, token_count = run_society(society)
            if answer is None:
                print(
                    f"Attempt {attempt + 1} returned None, chat_history: {chat_history}"
                )
                time.sleep(10)
                continue
            print(f"\033[94mAnswer: {answer}\033[0m")
            print(
                "Please verify the final answer using multiple tools (e.g., screenshots, webpage analysis)."
            )
            print(
                "CSV file 'alcohol_courses.csv' should be generated in the current directory."
            )
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(10)
            if attempt == max_retries - 1:
                print("All retries failed.")


if __name__ == "__main__":
    main()
