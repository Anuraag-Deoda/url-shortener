import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer, CircleModuleDrawer
from qrcode.image.styles.colormasks import SolidFillColorMask
from io import BytesIO
import base64
from PIL import Image
from typing import Optional
import os


class QRCodeService:
    """Service for generating QR codes for shortened URLs"""

    STYLES = {
        'default': {},
        'rounded': {'module_drawer': RoundedModuleDrawer()},
        'circles': {'module_drawer': CircleModuleDrawer()},
    }

    @staticmethod
    def generate_qr_code(
        url: str,
        size: int = 10,
        style: str = 'default',
        foreground_color: str = '#000000',
        background_color: str = '#FFFFFF',
        logo_path: Optional[str] = None
    ) -> BytesIO:
        """
        Generate a QR code image for the given URL.

        Args:
            url: The URL to encode
            size: Box size (1-40, default 10)
            style: 'default', 'rounded', or 'circles'
            foreground_color: Hex color for QR modules
            background_color: Hex color for background
            logo_path: Optional path to a logo image to embed

        Returns:
            BytesIO containing the PNG image
        """
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H if logo_path else qrcode.constants.ERROR_CORRECT_M,
            box_size=size,
            border=4,
        )

        qr.add_data(url)
        qr.make(fit=True)

        # Create image based on style
        if style != 'default' and style in QRCodeService.STYLES:
            style_opts = QRCodeService.STYLES.get(style, {})
            img = qr.make_image(
                image_factory=StyledPilImage,
                color_mask=SolidFillColorMask(
                    back_color=QRCodeService._hex_to_rgb(background_color),
                    front_color=QRCodeService._hex_to_rgb(foreground_color)
                ),
                **style_opts
            )
        else:
            img = qr.make_image(
                fill_color=foreground_color,
                back_color=background_color
            )

        # Add logo if provided
        if logo_path and os.path.exists(logo_path):
            img = QRCodeService._add_logo(img, logo_path)

        # Save to BytesIO
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)

        return buffer

    @staticmethod
    def generate_qr_base64(url: str, **kwargs) -> str:
        """Generate QR code and return as base64 data URL"""
        buffer = QRCodeService.generate_qr_code(url, **kwargs)
        base64_img = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{base64_img}"

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> tuple:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def _add_logo(qr_img, logo_path: str):
        """Add a logo to the center of the QR code"""
        logo = Image.open(logo_path)

        # Convert QR image to PIL Image if needed
        if hasattr(qr_img, 'get_image'):
            qr_img = qr_img.get_image()

        # Calculate logo size (max 30% of QR code)
        qr_width, qr_height = qr_img.size
        max_logo_size = int(min(qr_width, qr_height) * 0.3)

        # Resize logo maintaining aspect ratio
        logo.thumbnail((max_logo_size, max_logo_size), Image.Resampling.LANCZOS)

        # Calculate position
        logo_x = (qr_width - logo.width) // 2
        logo_y = (qr_height - logo.height) // 2

        # Paste logo
        if qr_img.mode != 'RGBA':
            qr_img = qr_img.convert('RGBA')
        if logo.mode != 'RGBA':
            logo = logo.convert('RGBA')

        qr_img.paste(logo, (logo_x, logo_y), logo)

        return qr_img
