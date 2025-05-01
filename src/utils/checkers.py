def compare_string_lists(list1, list2):
    set1 = set(list1)
    set2 = set(list2)

    only_in_list1 = list(set1 - set2)
    only_in_list2 = list(set2 - set1)
    in_both = list(set1 & set2)

    return only_in_list1, only_in_list2, in_both
