from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'build/my-languages.so',

  # Include one or more languages
  [
    'tree_sitter_libs/tree-sitter-java',
    'tree_sitter_libs/tree-sitter-kotlin',
    'tree_sitter_libs/tree-sitter-cpp'
  ]
)