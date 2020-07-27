from Code.Data.Text.Tokenisation.token_sequence import TokenSequence
from Code.Data.Text.Tokenisation.token_span_hierarchy import TokenSpanHierarchy
from Code.Data.Text.text import Text

example_text = """Weimar Republic is an unofficial, historical designation for the German state between 1919 and 1933. The name derives from the city of Weimar, where
its constitutional assembly first took place. The official name of the state was still "Deutsches Reich"; it had remained unchanged since 1871. In
English the country was usually known simply as Germany. A national assembly was convened in Weimar, where a new constitution for the "Deutsches
Reich" was written, and adopted on 11 August 1919. In its fourteen years, the Weimar Republic faced numerous problems, including hyperinflation,
political extremism (with paramilitaries  both left- and right-wing); and contentious relationships with the victors of the First World War. The
people of Germany blamed the Weimar Republic rather than their wartime leaders for the country's defeat and for the humiliating terms of the Treaty of
Versailles. However, the Weimar Republic government successfully reformed the currency, unified tax policies, and organized the railway system. Weimar
Germany eliminated most of the requirements of the Treaty of Versailles; it never completely met its disarmament requirements, and eventually paid
only a small portion of the war reparations (by twice restructuring its debt through the Dawes Plan and the Young Plan). Under the Locarno Treaties,
Germany accepted the western borders of the republic, but continued to dispute the Eastern border.""" * 1

example_token_sequence = TokenSequence(Text(example_text))

example_token_span_hierarchy = TokenSpanHierarchy(example_token_sequence)