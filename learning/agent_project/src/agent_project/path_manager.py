from pathlib import Path
import os

class PathManager:
    def __init__(self, data_root: str | Path | None = None):
        # 1.定位项目根目录(src-layout 下本文件位于 src/agent_project/,比原来多一层)
        self._curr_file = Path(__file__).resolve()
        self.PACKAGE_DIR = self._curr_file.parent       # src/agent_project(包自身目录)
        self.SRC_DIR = self.PACKAGE_DIR.parent          # src(源码根)
        self.PROJECT_ROOT = self.SRC_DIR.parent         # agent_project(项目根)

        # 2. 静态固定目录
        self.CONFIG_DIR = self.PROJECT_ROOT / "config"
        self.LOG_DIR = self.PROJECT_ROOT / "logs"

        # 3. 数据根目录优先级：外部传入 > 环境变量 > 默认项目内data
        if data_root is not None:
            self.DATA_ROOT = Path(data_root).resolve()
        else:
            env_data = os.getenv("APP_DATA_ROOT")
            if env_data:
                self.DATA_ROOT = Path(env_data).resolve()
            else:
                self.DATA_ROOT = (self.PROJECT_ROOT / "data").resolve()

        # 数据子目录
        self.RAW_DIR = self.DATA_ROOT / "raw"
        self.CLEANED_DIR = self.DATA_ROOT / "cleaned"
        self.CACHE_DIR = self.DATA_ROOT / "cache"
        self.OUTPUT_DIR = self.DATA_ROOT / "output"

        # 统一收纳所有需要创建的目录
        self.all_dirs = [
            self.CONFIG_DIR,
            self.LOG_DIR,
            self.DATA_ROOT,
            self.RAW_DIR,
            self.CLEANED_DIR,
            self.CACHE_DIR,
            self.OUTPUT_DIR,
        ]

    def init_all_dirs(self) -> None:
        """一次性创建全部目录，不存在自动生成"""
        for d in self.all_dirs:
            d.mkdir(parents=True, exist_ok=True)

    # 扩展工具示例：清空缓存目录
    def clear_cache(self):
        import shutil
        if self.CACHE_DIR.exists():
            shutil.rmtree(self.CACHE_DIR)
            self.CACHE_DIR.mkdir(parents=True)