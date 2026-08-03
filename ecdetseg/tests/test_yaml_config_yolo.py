import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engine.core.yaml_config import YAMLConfig  # noqa: E402


class YoloConfigValidationTest(unittest.TestCase):
    def test_rejects_accidental_double_equals_in_yolo_root(self):
        with self.assertRaisesRegex(ValueError, "single '='"):
            YAMLConfig.resolve_yolo_num_classes({
                'yolo_root': '=/home/ec2-user/dataset',
                'num_classes': 7,
            })

    def test_rejects_missing_yolo_root_even_with_explicit_num_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_root = Path(temp_dir) / 'missing-yolo-root'

            with self.assertRaisesRegex(FileNotFoundError, 'dataset root does not exist'):
                YAMLConfig.resolve_yolo_num_classes({
                    'yolo_root': str(missing_root),
                    'num_classes': 7,
                })

    def test_rejects_file_as_yolo_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root_file = Path(temp_dir) / 'dataset.txt'
            root_file.write_text('', encoding='utf-8')

            with self.assertRaisesRegex(NotADirectoryError, 'is not a directory'):
                YAMLConfig.resolve_yolo_num_classes({
                    'yolo_root': str(root_file),
                    'num_classes': 7,
                })

    def test_infers_num_classes_from_valid_yolo_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_file = root / 'data.yaml'
            data_file.write_text('names: [person, helmet, vest]\n', encoding='utf-8')
            cfg = {'yolo_root': str(root), 'num_classes': None}

            YAMLConfig.resolve_yolo_num_classes(cfg)

            self.assertEqual(cfg['num_classes'], 3)
            self.assertEqual(cfg['yolo_data_file'], str(data_file))

    def test_missing_data_file_explains_how_to_set_num_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, 'set `num_classes` explicitly'):
                YAMLConfig.resolve_yolo_num_classes({
                    'yolo_root': temp_dir,
                    'num_classes': None,
                })

    def test_valid_root_without_data_file_allows_explicit_num_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg = {'yolo_root': temp_dir, 'num_classes': 7}

            YAMLConfig.resolve_yolo_num_classes(cfg)

            self.assertEqual(cfg['num_classes'], 7)

    def test_rejects_missing_data_file_with_explicit_num_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_data_file = Path(temp_dir) / 'missing-data.yaml'

            with self.assertRaisesRegex(FileNotFoundError, 'data file does not exist'):
                YAMLConfig.resolve_yolo_num_classes({
                    'yolo_data_file': str(missing_data_file),
                    'num_classes': 7,
                })


if __name__ == '__main__':
    unittest.main()
