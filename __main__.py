from . import connect, MainWindow
from .main import main, parser

args = parser.parse_args()
connect(mock=args.debug)
main(MainWindow)
