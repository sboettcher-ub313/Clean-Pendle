import sys
import io

def inject_battle_log(log_text, template_path='battle_template.html', output_path='battle_arena.html'):
    """
    Injects the battle log into the HTML template and writes it to a new file.
    """
    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    injected = template.replace(
        '<!-- BATTLE_LOG_PLACEHOLDER -->',
        f'<script>\nconst battleLog = `{log_text}`;\n</script>'
    )

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(injected)

class BattleLogger:
    def __init__(self, to_file=None, js_file=None, inject_html=False, html_template='battle_template.html', html_output='battle_arena.html'):
        self.log_file = to_file
        self.js_file = js_file
        self.inject_html = inject_html
        self.html_template = html_template
        self.html_output = html_output
        self.buffer = io.StringIO()
        self.original_stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self.buffer
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        log = self.buffer.getvalue()
        self.buffer.close()

        # Save raw log
        if self.log_file:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write(log)

        # Write as JS file
        if self.js_file:
            escaped_log = log.replace("`", "\\`")
            js_content = f"const battleLog = `\n{escaped_log}\n`;"
            with open(self.js_file, "w", encoding="utf-8") as f:
                f.write(js_content)

        # Inject into HTML if needed
        if self.inject_html:
            try:
                inject_battle_log(log, self.html_template, self.html_output)
            except Exception as e:
                print(f"⚠️ HTML injection failed: {e}")

    def get_log(self):
        return self.buffer.getvalue()