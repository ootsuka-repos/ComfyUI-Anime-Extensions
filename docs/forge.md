# Doujin Forge nodes

`doujin-forge-core` のプラグインはワークフロー JSON を持ち、ComfyUI API に直接送信します。
推論や専用の Python 処理はこの拡張で実行し、別の Engine / Publisher API は起動しません。

| ノード | 入出力 |
| --- | --- |
| `ComfyUIExtensions.Forge.ComicPage` | IMAGE バッチを列数・余白・右読み順で1ページに配置 |
| `ComfyUIExtensions.Forge.VRMStarter` | 名前から技術確認用の VRM・ポスター・Blender シーンを出力 |
| `ComfyUIExtensions.Forge.VRMDance` | input 内の VRM と動画から RTMW の2D姿勢を抽出し、Blender で動画と編集用シーンを出力 |

VRM の Blender ブリッジと姿勢抽出は旧 Engine から移しました。
新しいコードは旧リポジトリを import しません。

VRM には ComfyUI ホスト側の Blender、VRM Add-on、ffmpeg が必要です。
Blender 実行ファイルは `COMFYUI_FORGE_BLENDER` で指定でき、既定は PATH の `blender` です。
VRM Add-on はその Blender のスクリプト検索先へインストールしてください。
RTMW は ComfyUI の既存 Python 環境で `rtmlib` と `onnxruntime` を使います。
プラグイン側に仮想環境やモデルは作りません。

入力ファイルは ComfyUI/input 内に限定します。結果は
`output/doujin-forge/avatar/<id>/` に保存し、history の `files` に取得用の記述を返します。
処理ログも同じフォルダに残します。取消時には Blender / ffmpeg を停止し、
姿勢抽出もフレーム間で取消を確認します。駆動動画に音声があれば出力動画へ合成します。

ワークフローの企画・分割・連結は呼び出し側エージェントが担当します。
ノード内から同じ ComfyUI キューへワークフローを送って完了待ちしないでください。
