# Contributing

Keep changes tied to a research question or a correctness requirement. Every new differentiable operation needs forward-reference and VJP checks, including broadcasting where relevant. Use finite differences away from nondifferentiable points. Kernel changes need a numerical gate before timing.

Run `python -m unittest discover -s tests -v`. Report optional skips. Add an engineering-log entry for substantive experiments and keep raw metrics; negative results are useful. Never replace planned measurements with invented numbers.

Code is currently an initial research baseline. Avoid API compatibility promises before backend and dtype contracts stabilize. Choose a license with the repository owner before external publication.
