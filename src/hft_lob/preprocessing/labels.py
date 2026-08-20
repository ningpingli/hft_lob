"""标签生成（需求文档 §7）：60 秒中间价对数/简单收益，future 匹配带容差、session 内。"""

from __future__ import annotations

import polars as pl

from hft_lob.configs.experiment import TargetConfig

#: 标签类型 → 列名短名（§7.1；``Target_<h>s_<short>`` 的推导来源）。
_LABEL_TYPE_SHORT: dict[str, str] = {"log_mid_return": "log", "simple_mid_return": "simple"}


def label_column(config: TargetConfig) -> str:
    """主标签列名（§7.1：一个实验唯一 primary target）。"""
    short = _LABEL_TYPE_SHORT[config.type]
    return f"Target_{config.horizon_seconds}s_{short}"


class LabelTransformer:
    """为清洗后的单日 DataFrame 追加未来中间价与标签列。

    行为契约（§2/§3/§7）：
    - 锚点 t = 当前快照时间；``y_t = return(mid_t, mid_future)``；
    - future 快照取 ``[t + h - tol, t + h + tol]`` 内最近一条（§7.2 容差匹配，
      禁止无上限 ``first timestamp >= t + h``）；
    - 标签只在 same trade_date AND same session_id 内构造：跨 session（如
      11:29:30 + 60s）与跨日自然得到 invalid（§3）；
    - 输出列：``future_mid``、``Target_<h>s_log``、``Target_<h>s_simple``
      （§7.1 主标签 + 对照）。
    """

    def __init__(self, config: TargetConfig) -> None:
        """初始化标签转换器。

        Args:
            config: 标签配置（类型 / 视界 / 容差）。
        """
        raise NotImplementedError("LabelTransformer.__init__ not implemented")

    def transform(self, df: pl.DataFrame) -> pl.DataFrame:
        """为清洗后的单日 DataFrame 追加未来中间价与标签列。

        Args:
            df: 含 ``trade_date / session_id / seconds / mid_price`` 的清洗后数据。

        Returns:
            追加 ``future_mid`` 与双标签列后的 DataFrame（invalid 标签为 null）。
        """
        raise NotImplementedError("LabelTransformer.transform not implemented")
