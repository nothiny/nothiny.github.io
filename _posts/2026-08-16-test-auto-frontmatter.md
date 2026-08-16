# 用 GitHub Actions 自动生成博客 front matter

写 Markdown 博客时,每篇都要手写 YAML 头(标题、日期、封面、标签),时间久了很繁琐。最近我把这件事交给了 GitHub Actions:我只写正文,构建时自动补全元信息。

## 实现思路

核心是一个 Node 脚本,在 `jekyll build` 之前扫描 `_posts` 目录。对于没有 front matter 的文章,脚本会从正文的第一个一级标题提取 `title`,从文件名提取 `date`,从正文第一张图片提取 `header-img`。

![流程图](img/in-posts/llm-inference.png)

## 交给 LLM 的部分

标签、副标题和 TL;DR 总结则调用 DeepSeek API 生成,结果按文章内容哈希缓存,文章没改动就不会重复调用。这样既省事,又不会因为 AI 挂了导致构建失败。

整体来说,这个方案让「写文章」这件事回到了最纯粹的样子:只需要关注内容本身。
