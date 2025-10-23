import flet as ft


def example():
    class Example(ft.Row):
        def __init__(self):
            super().__init__()
            self.pick_files_dialog = ft.FilePicker(on_result=self.pick_files_result)
            self.selected_files = ft.Text()

            def pick_files(_):
                self.pick_files_dialog.pick_files(allow_multiple=True)

            self.controls = [
                ft.ElevatedButton(
                    "Pick files",
                    icon=ft.Icons.UPLOAD_FILE,
                    on_click=pick_files,
                ),
                self.selected_files,
            ]

        def pick_files_result(self, e: ft.FilePickerResultEvent):
            self.selected_files.value = (
                ", ".join(map(lambda f: f.name, e.files)) if e.files else "Cancelled!"
            )
            self.selected_files.update()

        # happens when example is added to the page (when user chooses the FilePicker control from the grid)
        def did_mount(self):
            self.page.overlay.append(self.pick_files_dialog)
            self.page.update()

        # happens when example is removed from the page (when user chooses different control group on the navigation rail)
        def will_unmount(self):
            self.page.overlay.remove(self.pick_files_dialog)
            self.page.update()

    filepicker_example = Example()

    return filepicker_example


def main(page: ft.Page):
    page.title = "Flet counter example"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    input = ft.TextField(value="0", text_align=ft.TextAlign.RIGHT, width=100)

    def minus_click(e):
        input.value = str(int(input.value) - 1)
        page.update()

    def plus_click(e):
        input.value = str(int(input.value) + 1)
        page.update()

    page.add(
        ft.Row(
            alignment=ft.alignment.center,
            controls=[
                ft.IconButton(ft.Icons.REMOVE, on_click=minus_click),
                input,
                ft.IconButton(ft.Icons.ADD, on_click=plus_click),
            ],
        )
    )
    # page.add(example())


ft.run(main)
