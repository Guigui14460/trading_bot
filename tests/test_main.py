import unittest

from trading_bot.main import main


class MainTest(unittest.TestCase):
    def test_main(self):
        self.assertIsNone(main())


if __name__ == '__main__':
    unittest.main()
