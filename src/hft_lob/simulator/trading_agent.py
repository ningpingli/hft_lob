"""交易 Agent：维护多空头寸并记录交易历史。"""

from __future__ import annotations

from typing import Any


class Trading:
    """简单交易 Agent：以固定手数开平多空头寸并记录交易历史。"""

    def __init__(self, trading_hyperparameters: dict[str, Any]) -> None:
        """初始化交易 Agent。

        Args:
            trading_hyperparameters: trading 段超参数（本阶段仅存储）。
        """
        raise NotImplementedError("Trading.__init__ not implemented")

    def long(self, price: float, datetime: Any = None) -> None:  # noqa: ANN401
        """开多仓（手数 1）。

        Args:
            price: 成交价格。
            datetime: 成交时间（可选）。
        """
        raise NotImplementedError("Trading.long not implemented")

    def short(self, price: float, datetime: Any = None) -> None:  # noqa: ANN401
        """开空仓（手数 1）。

        Args:
            price: 成交价格。
            datetime: 成交时间（可选）。
        """
        raise NotImplementedError("Trading.short not implemented")

    def exit_long(self, price: float, datetime: Any = None) -> None:  # noqa: ANN401
        """平多仓并记录交易历史。

        Args:
            price: 成交价格。
            datetime: 成交时间（可选）。
        """
        raise NotImplementedError("Trading.exit_long not implemented")

    def exit_short(self, price: float, datetime: Any = None) -> None:  # noqa: ANN401
        """平空仓并记录交易历史。

        Args:
            price: 成交价格。
            datetime: 成交时间（可选）。
        """
        raise NotImplementedError("Trading.exit_short not implemented")
