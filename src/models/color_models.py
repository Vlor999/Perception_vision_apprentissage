"""All the class that define pixel color representation."""

from loguru import logger


class RGB:
    """Represent the RGB pixel format."""

    def __init__(self, R: int, G: int, B: int) -> None:
        """Init the RGB class."""
        self.R = R
        self.G = G
        self.B = B

    def to_HSV(self) -> "HSV":
        """Converting RGB to HSV."""
        logger.debug("Starting the conversion from RGB to HSV")
        R_prime = float(self.R) / 255
        G_prime = float(self.G) / 255
        B_prime = float(self.B) / 255

        C_max = max(R_prime, G_prime, B_prime)
        C_min = min(R_prime, G_prime, B_prime)

        delta = C_max - C_min

        if delta == 0:
            H = 0.0
        else:
            if C_max == R_prime:
                H = 60 * (((G_prime - B_prime) / delta) % 6)
            elif C_max == G_prime:
                H = 60 * (((B_prime - R_prime) / delta) + 2)
            elif C_max == B_prime:
                H = 60 * (((R_prime - G_prime) / delta) + 4)
        S = delta / C_max if C_max != 0 else 0
        hsv_value = HSV(H, S, C_max)
        logger.debug(
            f"End of conversion. From {self.__str__()}, to {hsv_value.__str__()}"
        )
        return hsv_value

    def __str__(self) -> str:
        """Return the st format of RGB."""
        return f"RGB: [{self.R}, {self.G}, {self.B}]"

    def __repr__(self) -> str:
        """Return the representation of RGB."""
        return f"[{self.R}, {self.G}, {self.B}]"

    def __eq__(self, value: object) -> bool:
        """Define the equal function for RGB."""
        if isinstance(value, RGB):
            return (value.G == self.G) and (value.G == self.G) and (value.B == self.B)
        return False


class HSV:
    """Represent the HSV pixel format."""

    def __init__(self, H: float, S: float, V: float) -> None:
        """Init HSV value."""
        self.H = H
        self.S = S
        self.V = V

    def to_RGB(self) -> "RGB":
        """Convert RGB to HSV."""
        logger.debug("Starting the conversion from HSV to RGB")
        C = self.V * self.S
        X = C * (1 - abs((self.H / 60) % 2 - 1))
        m = self.V - C
        if 0 <= self.H < 60:
            R_prime, G_prime, B_prime = (C, X, 0.0)
        elif 60 <= self.H < 120:
            R_prime, G_prime, B_prime = (X, C, 0.0)
        elif 120 <= self.H < 180:
            R_prime, G_prime, B_prime = (0.0, C, X)
        elif 180 <= self.H < 240:
            R_prime, G_prime, B_prime = (0.0, X, C)
        elif 240 <= self.H < 300:
            R_prime, G_prime, B_prime = (X, 0.0, C)
        else:
            R_prime, G_prime, B_prime = (C, 0.0, X)
        R, G, B = ((R_prime + m) * 255, (G_prime + m) * 255, (B_prime + m) * 255)
        rgb_value = RGB(round(R), round(G), round(B))
        logger.debug(
            f"End of conversion. From {self.__str__()}, to {rgb_value.__str__()}"
        )
        return rgb_value

    def __eq__(self, value: object) -> bool:
        """Define the equal funtion of HSV."""
        if isinstance(value, HSV):
            return (value.H == self.H) and (value.S == self.S) and (value.V == self.V)
        return False

    def __str__(self) -> str:
        """Return the string conversiuon of HSV."""
        return f"HSV: [{self.H}, {self.S}, {self.V}]"

    def __repr__(self) -> str:
        """Return the representation of HSV."""
        return f"[{self.H}, {self.S}, {self.V}]"
