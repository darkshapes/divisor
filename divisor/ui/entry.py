from pathlib import Path
from textual import events, on
from textual.app import App, ComposeResult
from textual.widgets import Footer, Label, Tabs
from nnll.constants import ExtensionType


class TabsApp(App):
    """Demonstrates the Tabs widget."""

    CSS = """
    Tabs {
        dock: top;
    }
    Screen {
        align: center middle;
    }
    Label {
        margin:1 1;
        width: 100%;
        height: 100%;
        background: $panel;
        border: tall $primary;
        content-align: center middle;
    }
    """

    BINDINGS = [
        ("a", "add", "Add tab"),
        ("r", "remove", "Remove active tab"),
        ("c", "clear", "Clear tabs"),
    ]

    def get_all_webp_files(self, output_dir: str = ".output") -> list[str]:
        """Get all WebP filenames from .output directory, removing file extensions.
        Returns all files in reverse order."""
        webp_ext = next(iter(ExtensionType.WEBP))
        names = []
        output_path = Path(output_dir)
        if output_path.exists() and output_path.is_dir():
            for file_path in output_path.iterdir():
                if file_path.is_file() and file_path.suffix == webp_ext:
                    # Remove the suffix (file extension)
                    name = file_path.stem.replace("divisor_", " ")
                    names.append(name)
        # Sort and reverse to get newest files first
        return sorted(names, reverse=True)

    def get_names_from_output(self, offset: int = 0, limit: int = 10) -> list[str]:
        """Get a slice of filenames from the cached list."""
        return self.all_webp_files[offset : offset + limit]

    def compose(self) -> ComposeResult:
        # Load all WebP files and cache them
        self.all_webp_files: list[str] = self.get_all_webp_files()
        self.loaded_count: int = 0  # Track how many tabs have been loaded

        # Load first 10 tabs
        self.tab_titles: list[str] = self.get_names_from_output(0, 10)
        self.loaded_count = len(self.tab_titles)

        initial_tab = self.tab_titles[0] if self.tab_titles else ""
        yield Tabs(initial_tab)
        yield Label()
        yield Footer()

    def on_mount(self) -> None:
        """Focus the tabs when the app starts."""
        tabs = self.query_one(Tabs)
        tabs.focus()
        # Add remaining tabs in reverse order (newest first)
        for title in self.tab_titles[1:]:
            tabs.add_tab(title)

    def on_tabs_tab_activated(self, event: Tabs.TabActivated) -> None:
        """Handle TabActivated message sent by Tabs."""
        label = self.query_one(Label)
        if event.tab is None:
            # When the tabs are cleared, event.tab will be None
            label.visible = False
        else:
            label.visible = True
            label.update(event.tab.label)

            # Check if the last tab was activated
            if self.tab_titles and event.tab.label == self.tab_titles[-1]:
                # Load next 10 files if available
                next_batch = self.get_names_from_output(self.loaded_count, 10)
                if next_batch:
                    tabs = self.query_one(Tabs)
                    for title in next_batch:
                        tabs.add_tab(title)
                        self.tab_titles.append(title)
                    self.loaded_count += len(next_batch)

    def action_add(self) -> None:
        """Add a new tab."""
        tabs = self.query_one(Tabs)
        # Cycle the names
        self.tab_titles[:] = [*self.tab_titles[1:], self.tab_titles[0]]
        tabs.add_tab(self.tab_titles[0])

    def action_remove(self) -> None:
        """Remove active tab."""
        tabs = self.query_one(Tabs)
        active_tab = tabs.active_tab
        if active_tab is not None:
            tabs.remove_tab(active_tab.id)

    def action_clear(self) -> None:
        """Clear the tabs."""
        self.query_one(Tabs).clear()

    @on(events.Enter)
    def on_enter(self, event: events.Enter) -> None:
        """Force terminal mouse event monitoring"""
        tabs = self.query_one(Tabs)
        if event.node == tabs or event.node == self:
            self.hover = True

    @on(events.Leave)
    def on_leave(self, event: events.Leave) -> None:
        """Force terminal mouse event monitoring"""
        tabs = self.query_one(Tabs)
        if event.node == tabs or event.node == self:
            self.hover = True

    def action_show_tab(self, tab_id: str) -> None:
        """Show the tab."""
        tabs = self.query_one(Tabs)
        tabs.get_child_by_type(Tabs).active

    @on(events.MouseScrollDown)
    def on_mouse_down(self, event: events.MouseDown) -> None:
        """Joever"""
        tabs = self.query_one(Tabs)
        if tabs.has_focus:
            tabs.action_next_tab()

    @on(events.MouseScrollUp)
    def on_mouse_up(self, event: events.MouseUp) -> None:
        """NOT Joever"""
        tabs = self.query_one(Tabs)
        if tabs.has_focus and tabs.active != "tab-1":  # Don't wrap
            tabs.action_previous_tab()


if __name__ == "__main__":
    app = TabsApp()
    app.run()
