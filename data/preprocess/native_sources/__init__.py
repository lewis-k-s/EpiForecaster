"""Native-support preprocessing builders for heterogeneous source graphs.

Each module here builds one offline artifact under
``data/processed/native_support/`` per the frozen schemas in
``plan_heterogeneous_graph_refactor.md`` §11.3. Source inputs are read-only;
never modify ``data/files/`` or sibling project checkouts in place.
"""
