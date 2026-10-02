import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bio_hash import FlyBioHash


class TestBioHash(unittest.TestCase):
    def test_avalanche_effect(self):
        hasher = FlyBioHash()
        h1 = hasher.hash(b"The quick brown fox jumps over the lazy dog")
        h2 = hasher.hash(b"The quick brown fox jumps over the lazy dog.")
        b1 = bin(int(h1, 16))[2:].zfill(256)
        b2 = bin(int(h2, 16))[2:].zfill(256)
        hamming = sum(c1 != c2 for c1, c2 in zip(b1, b2))
        self.assertTrue(100 <= hamming <= 160, f"Avalanche failed: {hamming} bits flipped")


if __name__ == "__main__":
    unittest.main()
