"""
churn_gui.py
=============
CodeQuest Arena | Churn Rate Predictor v2.0 — Interface Gráfica

Uma janela simples (Tkinter, já vem com o Python — não precisa instalar
nada além do que o pipeline já usa) que executa o churn_pipeline.py e
mostra em tempo real, na tela, tudo o que normalmente apareceria no
console: limpeza dos dados, métricas dos modelos, matriz de confusão,
variáveis mais importantes e a confirmação de que o modelo foi salvo.

Como executar:
    python churn_gui.py

Requisitos:
    - Os arquivos churn_pipeline.py e telco_churn.csv devem estar na
      MESMA pasta que este arquivo (rode generate_dataset.py antes,
      se ainda não tiver gerado o telco_churn.csv).
"""

import io
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext

import churn_pipeline  # reaproveita as funções já prontas do pipeline


class RedirecionadorDeSaida(io.StringIO):
    """
    Trecho de suporte: intercepta tudo que seria escrito no console
    (via print) e manda para o widget de texto da janela, em vez do
    terminal. É assim que a GUI consegue "espelhar" a saída do
    pipeline sem precisar reescrever nenhuma função dele.
    """

    def __init__(self, text_widget: scrolledtext.ScrolledText):
        super().__init__()
        self.text_widget = text_widget

    def write(self, texto: str):
        # Insere o texto no widget e rola automaticamente para o final
        self.text_widget.after(0, self._inserir, texto)
        return len(texto)

    def _inserir(self, texto: str):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, texto)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state="disabled")

    def flush(self):
        pass


class ChurnApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CodeQuest Arena — Churn Rate Predictor v2.0")
        self.root.geometry("820x560")
        self.root.configure(bg="#1e1e1e")

        # --- Cabeçalho ---
        titulo = tk.Label(
            root,
            text="Previsor de Churn Rate",
            font=("Segoe UI", 16, "bold"),
            bg="#1e1e1e",
            fg="#ffffff",
        )
        titulo.pack(pady=(15, 0))

        subtitulo = tk.Label(
            root,
            text="Pandas + Scikit-Learn | Regressão Logística vs Random Forest",
            font=("Segoe UI", 10),
            bg="#1e1e1e",
            fg="#9a9a9a",
        )
        subtitulo.pack(pady=(0, 10))

        # --- Botão de execução ---
        self.botao_rodar = tk.Button(
            root,
            text="▶  Executar Pipeline",
            font=("Segoe UI", 11, "bold"),
            bg="#2e7d32",
            fg="white",
            activebackground="#1b5e20",
            relief="flat",
            padx=16,
            pady=8,
            command=self.rodar_pipeline,
        )
        self.botao_rodar.pack(pady=5)

        # --- Barra de status ---
        self.status_var = tk.StringVar(value="Pronto para executar.")
        status_label = tk.Label(
            root,
            textvariable=self.status_var,
            font=("Segoe UI", 9, "italic"),
            bg="#1e1e1e",
            fg="#7fbf7f",
        )
        status_label.pack(pady=(0, 5))

        # --- Área de log (espelha o console) ---
        self.area_log = scrolledtext.ScrolledText(
            root,
            font=("Consolas", 10),
            bg="#0d1117",
            fg="#c9d1d9",
            insertbackground="white",
            state="disabled",
            wrap="word",
        )
        self.area_log.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def rodar_pipeline(self):
        # Desabilita o botão durante a execução para evitar cliques duplos
        self.botao_rodar.configure(state="disabled", text="Executando...")
        self.status_var.set("Executando pipeline, aguarde...")

        # Limpa o log anterior
        self.area_log.configure(state="normal")
        self.area_log.delete("1.0", tk.END)
        self.area_log.configure(state="disabled")

        # Roda em uma thread separada para a janela não "congelar"
        # enquanto o modelo treina.
        thread = threading.Thread(target=self._executar_em_thread, daemon=True)
        thread.start()

    def _executar_em_thread(self):
        redirecionador = RedirecionadorDeSaida(self.area_log)
        saida_original = sys.stdout
        sys.stdout = redirecionador  # a partir daqui, todo print() vai pra janela

        try:
            df = churn_pipeline.load_and_clean(churn_pipeline.CSV_PATH)
            df = churn_pipeline.encode_categoricals(df)
            melhor_modelo, melhor_nome = churn_pipeline.train_and_evaluate(df)
            churn_pipeline.save_model(melhor_modelo, melhor_nome)
            self.root.after(0, self._ao_concluir, True, melhor_nome)
        except FileNotFoundError:
            self.root.after(0, self._ao_concluir, False,
                             "Arquivo telco_churn.csv não encontrado. "
                             "Rode generate_dataset.py primeiro.")
        except Exception as e:
            self.root.after(0, self._ao_concluir, False, str(e))
        finally:
            sys.stdout = saida_original  # restaura o console normal

    def _ao_concluir(self, sucesso: bool, mensagem: str):
        self.botao_rodar.configure(state="normal", text="▶  Executar Pipeline")
        if sucesso:
            self.status_var.set(f"Concluído! Melhor modelo: {mensagem}")
        else:
            self.status_var.set("Erro na execução.")
            messagebox.showerror("Erro ao executar o pipeline", mensagem)


if __name__ == "__main__":
    root = tk.Tk()
    app = ChurnApp(root)
    root.mainloop()
