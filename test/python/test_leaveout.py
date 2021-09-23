import unittest
from zensols.persist import ReadOnlyStash
from zensols.dataset import DatasetError, LeaveNOutSplitKeyContainer


class RangeStash(ReadOnlyStash):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.prefix = ''

    def load(self, name: str):
        return f'{self.prefix}{name}'

    def keys(self):
        return map(str, range(self.n))


class TestLeaveNOutSplitKeyContainer(unittest.TestCase):
    def test_splits(self):
        shoulds = [
            {'validation': ('0',), 'test': ('1',), 'train': ('2', '3', '4')},
            {'validation': ('1',), 'test': ('2',), 'train': ('3', '4', '0')},
            {'validation': ('2',), 'test': ('3',), 'train': ('4', '0', '1')},
            {'validation': ('3',), 'test': ('4',), 'train': ('0', '1', '2')},
            {'validation': ('4',), 'test': ('0',), 'train': ('1', '2', '3')}
        ]
        rs = RangeStash(5)
        c = LeaveNOutSplitKeyContainer(rs, shuffle=False)
        self.assertEqual({'train', 'test', 'validation'}, c.split_names)
        self.assertEqual({'train': 3, 'test': 1, 'validation': 1},
                         c.counts_by_key)
        for i in range(len(shoulds) * 3):
            ix = i % len(shoulds)
            should_reset = ((i+1) % len(shoulds)) == 0
            should = shoulds[ix]
            self.assertEqual(should, c.keys_by_split)
            reset = c.next_split()
            self.assertEqual(should_reset, reset)

    def test_split_config(self):
        rs = RangeStash(5)
        dist = {'train': -1, 'validation': 2, 'test': 1}
        c = LeaveNOutSplitKeyContainer(rs, shuffle=False, distribution=dist)
        self.assertEqual({'train', 'test', 'validation'}, c.split_names)
        self.assertEqual({'train': 2, 'test': 1, 'validation': 2},
                         c.counts_by_key)

    def test_too_many_takers(self):
        rs = RangeStash(5)
        dist = {'train': -1, 'validation': -1, 'test': 1}
        c = LeaveNOutSplitKeyContainer(rs, shuffle=False, distribution=dist)
        self.assertEqual({'train', 'test', 'validation'}, c.split_names)
        with self.assertRaisesRegex(DatasetError, '^Distribution has more'):
            c.keys_by_split

    def test_bad_dist(self):
        rs = RangeStash(5)
        dist = {'train': 1, 'validation': 1, 'test': 1}
        c = LeaveNOutSplitKeyContainer(rs, shuffle=False, distribution=dist)
        self.assertEqual({'train', 'test', 'validation'}, c.split_names)
        msg = (r'^Number of allocated keys to the distribution \(3\) ' +
               r'does not equal total keys \(5\)')
        with self.assertRaisesRegex(DatasetError, msg):
            c.keys_by_split
