#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import random


# class of functions of different weight sharing mappings
class WeightSharingTopology:
    """Class of predefined functions of different sharing structures."""

    def uniform_sequential(share_rate, num_share):
        mapping = []
        for i in range(num_share):
            for _ in range(share_rate):
                mapping.append(i)
        return mapping

    def uniform_strided(share_rate, num_share):
        mapping = []
        for _ in range(share_rate):
            for i in range(num_share):
                mapping.append(i)
        return mapping

    def uniform_random(share_rate, num_share):
        mapping = WeightSharingMapping.scatter_seq(share_rate, num_share)
        random.shuffle(mapping)
        return mapping
