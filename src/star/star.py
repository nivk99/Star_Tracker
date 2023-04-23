class Star:

    def __init__(self, ID: int = -1, x: float = 0, y: float = 0, r: float = 0, b: float = 0):
        self._ID = ID
        self._r = r
        self._b = b
        self._y = y
        self._x = x

    def get_x(self) -> float:
        return self._x

    def get_y(self) -> float:
        return self._y

    def get_b(self) -> float:
        return self._b

    def get_r(self) -> float:
        return self._r

    def get_id(self) -> int:
        return self._ID

    def set_x(self, x: float = 0) -> None:
        self._x = x

    def set_y(self, y: float = 0) -> None:
        self._y = y

    def set_b(self, b: float = 0) -> None:
        self._b = b

    def set_r(self, r: float = 0) -> None:
        self._r = r

    def set_id(self, ID: int = -1) -> None:
        self._ID = ID

    def __str__(self):
        return f"id:{self._ID}, x:{self._x}, y:{self._y}, b:{self._b}, r:{self._r}"

    def __repr__(self):
        return f"(id:{self._ID},x:{self._x},y:{self._y},b:{self._b},r:{self._r})"
