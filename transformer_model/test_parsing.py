from tree_sitter import Language, Parser

JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
KOTLIN_LANGUAGE = Language('build/my-languages.so', 'kotlin')
CPP_LANGUAGE = Language('build/my-languages.so', 'cpp')

parser = Parser()
parser.set_language(JAVA_LANGUAGE)


sample_code = """import java.io.BufferedReader;\nimport java.io.IOException;\nimport java.io.InputStream;\nimport java.io.InputStreamReader;\nimport java.util.InputMismatchException;\nimport java.util.StringTokenizer;\n\npublic class Main {\n    public static void main(String[] args) {\n        QuickReader cin = new QuickReader(System.in);\n        int n = cin.nextInt();\n        String[] a = new String[3];\n        for (int i = 0; i < 3; i++) {\n            a[i] = cin.next();\n        }\n        Solution solution = new Solution(a, n);\n        int q = cin.nextInt();\n        StringBuilder sb = new StringBuilder();\n        for (int i = 0; i < q; i++) {\n            sb.append(solution.func(cin.nextInt() - 1, cin.nextInt() - 1));\n            sb.append(""\n"");\n        }\n        System.out.println(sb);\n    }\n}\n\nclass Solution {\n    int[] arr;\n    int[][] d;\n    int[] sums;\n\n    public Solution(String[] a, int n) {\n        arr = new int[n + 1];\n        d = new int[2][n + 1];\n        for (int i = 0; i < n; i++) {\n            for (int j = 0; j < 3; j++) {\n                arr[i + 1] += (a[j].charAt(i) - '0') << j;\n            }\n        }\n        sums = new int[n + 1];\n        boolean c = false;\n        for (int i = 0; i < n; i++) {\n            sums[i + 1] = sums[i];\n            int b = arr[i + 1];\n            int b1 = arr[i];\n            if (b == 1 || b == 2 || b == 3 || b == 4 || b == 6) {\n                if ((b & b1) == 0) {\n                    sums[i + 1]++;\n                }\n            } else if (b == 5) {\n                if (b1 == 0 || b1 == 2) {\n                    sums[i + 1] += 2;\n                    c = false;\n                } else if (b1 == 1 || b1 == 3 || b1 == 4 || b1 == 6) {\n                    sums[i + 1]++;\n                    c = false;\n                }\n            } else if (b == 7) {\n                if (b1 == 0) {\n                    sums[i + 1]++;\n                } else if (b1 == 5 && !c) {\n                    sums[i + 1]--;\n                }\n                c = true;\n            }\n        }\n        int p = 0;\n        for (int i = 1; i < arr.length; i++) {\n            if (arr[i] == 7) {\n                p = i;\n            } else if (arr[i] == 5) {\n                d[0][i] = p;\n            } else {\n                p = 0;\n            }\n        }\n        p = 0;\n        for (int i = arr.length - 1; i >= 1; i--) {\n            if (arr[i] == 7) {\n                p = i;\n            } else if (arr[i] == 5) {\n                d[1][i] = p;\n            } else {\n                p = 0;\n            }\n        }\n    }\n\n    public int func(int l, int r) {\n        int b = arr[l + 1];\n        int b1 = arr[l];\n        if (l == r) {\n            if (b == 5) {\n                return 2;\n            } else if (b == 0) {\n                return 0;\n            }\n            return 1;\n        }\n        int c = sums[r + 1] - sums[l];\n        if (b == 1 || b == 2 || b == 3 || b == 4 || b == 6) {\n            if ((b & b1) != 0) {\n                c++;\n            }\n        } else if (b == 7) {\n            if (b1 != 0 && b1 != 5) {\n                c++;\n            } else if (b1 == 5 && d[0][l] == 0) {\n                c += 2;\n            } else if (b1 == 5) {\n                c++;\n            }\n        } else if (b == 5) {\n            if (b1 == 5 || b1 == 7) {\n                c += 2;\n                if (d[1][l + 1] != 0 && d[1][l + 1] <= r + 1 && d[0][l + 1] != 0) {\n                    c--;\n                }\n            } else if (b1 == 1 || b1 == 3 || b1 == 4 || b1 == 6) {\n                c++;\n            }\n        }\n        return c;\n    }\n}\n\nclass QuickReader {\n    BufferedReader in;\n    StringTokenizer token;\n\n    public QuickReader(InputStream ins) {\n        in = new BufferedReader(new InputStreamReader(ins));\n        token = new StringTokenizer("""");\n    }\n\n    public boolean hasNext() {\n        while (!token.hasMoreTokens()) {\n            try {\n                String s = in.readLine();\n                if (s == null) {\n                    return false;\n                }\n                token = new StringTokenizer(s);\n            } catch (IOException e) {\n                throw new InputMismatchException();\n            }\n        }\n        return true;\n    }\n\n    public String next() {\n        hasNext();\n        return token.nextToken();\n    }\n\n    public int nextInt() {\n        return Integer.parseInt(next());\n    }\n\n    public int[] nextInts(int n) {\n        int[] res = new int[n];\n        for (int i = 0; i < n; i++) {\n            res[i] = nextInt();\n        }\n        return res;\n    }\n\n    public long nextLong() {\n        return Long.parseLong(next());\n    }\n\n    public long[] nextLongs(int n) {\n        long[] res = new long[n];\n        for (int i = 0; i < n; i++) {\n            res[i] = nextLong();\n        }\n        return res;\n    }\n}"""

tree = parser.parse(bytes(sample_code, "utf-8"))

# Inspect resultant tree object
root_node = tree.root_node
print(root_node.sexp())