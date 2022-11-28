from . import build_root
from .pydrive_utils import setup
from .main import main, parser

args = parser.parse_args()
setup(debug=args.debug)
main(build_root)

