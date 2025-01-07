import base64
import io

import matplotlib as mpl


# Helper function to convert a matplotlib figure into an HTML string
def _convert_fig_into_html(fig: mpl.figure.Figure) -> str:
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="png")
    img_base64 = base64.b64encode(img_buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{img_base64}" alt="Rendered Image" />'
