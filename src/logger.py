from rich.console import Console


def working_on(message):
  console = Console()
  console.print(":wrench: [bold green]WORKING ON[/bold green]: " + message)


def warning(message):
  console = Console()
  console.print(":tomato: [bold red]WARNING[/bold red]: " + message)


def info(message):
  console = Console()
  console.print(
      ":information_source: [bold yellow]INFO[/bold yellow]: " + message)


if __name__ == "__main__":
  warning("This is a warning message")
  working_on("This is a working on message")
  info("This is an info message")
