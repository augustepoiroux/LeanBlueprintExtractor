from collections import defaultdict
from typing import Any

from lean_blueprint_extractor.utils import logger


def longest_common_suffix_with_dots(str1: str, str2: str) -> str:
    # Split the strings by dots
    parts1 = str1.split(".")
    parts2 = str2.split(".")

    # Reverse the lists of parts
    parts1.reverse()
    parts2.reverse()

    # Find the longest common prefix of the reversed lists
    common_parts = []
    for part1, part2 in zip(parts1, parts2):
        if part1 == part2:
            common_parts.append(part1)
        else:
            break

    # Reverse the common parts to get the longest common suffix
    common_suffix = ".".join(common_parts[::-1])
    return common_suffix


def find_decl_info(
    decl_name: str, declarations: dict[str, list[dict[str, Any]]], short_name_to_decl: dict[str, list[str]]
):
    if decl_name in declarations:
        return declarations[decl_name], decl_name
    else:
        # try to find a match by discarding everything before the last dot
        short_decl_name = decl_name.split(".")[-1]
        if short_decl_name in short_name_to_decl:
            if len(short_name_to_decl[short_decl_name]) == 1:
                decl_name = short_name_to_decl[short_decl_name][0]
                return declarations[decl_name], decl_name
            else:
                # we have to refine the search by finding the match with the longest common suffix
                matches = []
                suffix_len = 0
                largest_common_suffix = ""
                for full_decl_name in short_name_to_decl[short_decl_name]:
                    local_common_suffix = longest_common_suffix_with_dots(decl_name, full_decl_name)
                    if len(local_common_suffix) > suffix_len:
                        matches = [full_decl_name]
                        suffix_len = len(local_common_suffix)
                        largest_common_suffix = local_common_suffix
                    elif len(local_common_suffix) == suffix_len:
                        matches.append(full_decl_name)
                if len(matches) == 1:
                    return declarations[matches[0]], matches[0]
                else:
                    logger.warning("Multiple matches for %s found: %s", decl_name, matches)
                    res = []
                    for match in matches:
                        res.extend(declarations[match])
                    return res, largest_common_suffix
        else:
            logger.info("No match found for %s", decl_name)
    return None, None


def merge_blueprint_lean_dep_graphs(blueprint_extracted: list[dict], lean_declarations: list[dict]) -> list[dict]:
    if len({decl["full_name"] for decl in lean_declarations}) != len(lean_declarations):
        logger.warning("Lean declarations with duplicate names found")

    lean_declarations_dict: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decl in lean_declarations:
        lean_declarations_dict[decl["full_name"]].append(decl)
    short_name_to_decl = defaultdict(list)
    for decl_name in lean_declarations_dict:
        short_name = decl_name.split(".")[-1]
        short_name_to_decl[short_name].append(decl_name)

    # Get lean declarations
    for attributes in blueprint_extracted:
        if "leandecls" in attributes:
            leandecls = attributes.pop("leandecls")
            attributes["lean_names"] = leandecls
            attributes["lean_declarations"] = []
            for leandecl in leandecls:
                decl_info_list, _ = find_decl_info(leandecl, lean_declarations_dict, short_name_to_decl)
                if decl_info_list is None:
                    attributes["lean_declarations"].append({"full_name": leandecl})
                else:
                    for decl_info in decl_info_list:
                        if decl_info is not None:
                            attributes["lean_declarations"].append({"full_name": leandecl, **decl_info})
                        else:
                            attributes["lean_declarations"].append({"full_name": leandecl})

    return blueprint_extracted
