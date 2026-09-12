# YuE2（ComfyUI のローカル音楽生成）

`YuE2Generate` は本リポジトリの実ノードです。ComfyUI のキューで実行し、
`AUDIO` と `truncated` を返し、元の 24-bit FLAC をプレビューできます。
外部の YuE2 HTTP サービスは使用しません。Torch/Transformers の依存衝突を避けるため、
YuE2 の推論だけを独立 Python 子プロセスで実行します。ComfyUI の中断で子も停止し、
終了時にはモデルを解放します。共有 ComfyUI の Python に YuE2 をインストールしないでください。

## 必要なローカル環境

- 上流: https://github.com/multimodal-art-projection/YuE
- 動作確認: YuE2 inference 0.1.6、上流コード `88da114a67df892af0329472073b96a5ef700b93`。
- 独立 Python 3.12、Torch 2.10.0、Transformers 4.57.6、CUDA BF16。
- YuE2-3B: `29b3558dd46954a0cd9021dc76d5c91864a0f1c7`。
- YuE2-Vae: `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a`。
- モデルは事前配置が必要です。ノードはダウンロード・外部送信をせず、
  `local_files_only=True` とオフライン環境で実行します。上流が重みのマニフェストを検証します。

`ComfyUI/runtimes/YuE2/comfyui.json` をローカルに作成してください。
別の場所は ComfyUI 起動環境の `COMFYUI_YUE2_CONFIG` で指定できます。
この設定は管理者専用で、ノード入力から実行ファイル・出力パスを受け取りません。

```json
{
  "python": "/absolute/path/to/YuE2/.venv/bin/python",
  "model": "/absolute/path/to/YuE2-3B/snapshot",
  "vae": "/absolute/path/to/YuE2-Vae/snapshot"
}
```

ComfyUI を処理待ち・実行中のジョブがない時に再起動すると登録されます。
`GET /object_info/YuE2Generate` で登録、`GET /yue2/status` で interpreter と重みの
配置・サイズを確認できます。後者はモデルロード成功や音楽品質の保証ではありません。
モデルはジョブごとに読み込み、常駐しないため `model_loaded` は false です。

## 入力と出力

- `style` / `lyrics`: 別々に与え、改行を含めてそのまま上流へ渡します。
- `cot`: `full`（既定）、`melody`、`off`。`abc` は任意の上流スコア条件です。
- `cfg_scale`: `-1` は上流のモード別既定値（None）です。独自のCFG既定値を上書きしません。
- `seed`: 既定 42。既定の非量子化・32-step ODE・生成上限を変更しません。
- `noncommercial`: 既定 false。**CC-BY-NC-4.0 の非商用利用に明示的に同意した時だけ true**。

重みは Multimodal Art Projection / YuE2 authors の **CC BY-NC 4.0** です。
上流コードの Apache 2.0 と混同しないでください。日本語歌唱品質は未検証です。

`ComfyUI/output/yue2/<random-id>/` に入力 `intent.json`、`worker.log`、
`completion.json` と `native/` を保存します。`native/` には元の音源、ABC、
トークン、latent、request/config、重みの識別情報・ハッシュを含む result が残ります。
非有限・空・無音のモデル出力は失敗にします。`truncated` は ABC と semantic の論理和です。
再生可能でも true ならフル楽曲の完成とは扱わないでください。

ComfyUI API の通常の `/prompt` → `/history/{prompt_id}` → `/view` を利用します。
ComfyUI のメモリ履歴は再起動で消える場合があります。クライアントは完了結果を保存し、
履歴不明時に自動再生成しないでください。元の音源・native artifacts は削除しません。

## テスト

ComfyUI の Python で `tests/test_yue2.py` を実行できます。推論部分を置き換える
ユニットテストであり、本物のモデル生成の証拠ではありません。
既存のテストが `sys.modules` を書き換えるため、テストファイルは別プロセスで実行します。
