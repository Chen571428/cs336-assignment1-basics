#
# Much more slower...
import cppyy
cppyy.cppdef(r"""
#include <string>
struct Byte_Pair_Counts_Obj
{
    std::string s1;
    std::string s2;

    Byte_Pair_Counts_Obj(std::string a, std::string b) : s1(a), s2(b) {}

    bool operator<(const Byte_Pair_Counts_Obj& other) const {
        if (s1 != other.s1) return s1 > other.s1;
        return s2 > other.s2;
    }

    bool operator==(const Byte_Pair_Counts_Obj& other) const {
        return s1 == other.s1 && s2 == other.s2;
    }
};
""")

Byte_Pair_Counts_Obj = cppyy.gbl.Byte_Pair_Counts_Obj