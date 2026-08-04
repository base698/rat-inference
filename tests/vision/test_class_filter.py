"""target_class_matches: the filter that silently discarded every detection
when the trained class name ('item', from single-class training) didn't
match the CLI filter ('redbull'). Pins the gotcha."""
from ratbot.vision.yolo_inference import target_class_matches


def test_exact_name_matches():
    assert target_class_matches("item", 0, "item")
    assert target_class_matches("rat", 0, "rat")


def test_substring_match_is_allowed():
    assert target_class_matches("red bull can", 0, "red bull")


def test_the_item_vs_redbull_gotcha_does_not_match():
    # single-class training names the class 'item'; filtering on the dataset
    # yaml name finds nothing -> turret never locks on. Document it forever.
    assert not target_class_matches("item", 0, "redbull")


def test_class_index_string_matches():
    assert target_class_matches("whatever", 3, "3")


def test_generated_alias_matches():
    assert target_class_matches("class0", 0, "class_0")
