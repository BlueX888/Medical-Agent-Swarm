"""
网络搜索模块

提供医学领域的网络搜索能力，基于 DuckDuckGo Search API (DDGS)
"""
from typing import List, Dict, Any, Optional, Iterable, Tuple
from dataclasses import dataclass
from loguru import logger
import asyncio
import os
import re
from bs4 import BeautifulSoup
import httpx
from dotenv import load_dotenv

load_dotenv()

# 导入 DDGS：优先使用新包名 ddgs（v9+），降级到旧包名 duckduckgo_search
try:
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.error("DDGS not available. Install with: pip install ddgs")


@dataclass
class SearchResult:
    """搜索结果数据结构"""
    title: str
    url: str
    snippet: str  # 摘要
    source: str = "web"  # 来源标识


class SearchResults(list):
    """List-compatible search result collection carrying retry telemetry."""

    def __init__(
        self,
        values: Optional[Iterable[SearchResult]] = None,
        *,
        retry_count: int = 0,
        retry_exhausted: bool = False,
    ):
        super().__init__(values or [])
        self.retry_count = max(0, int(retry_count))
        self.retry_exhausted = bool(retry_exhausted)


class WebSearchTool:
    """
    网络搜索工具

    功能：
    - 使用 DuckDuckGo 进行网络搜索
    - 专注于医学领域网站
    - 结果去重和质量过滤
    """

    MEDICAL_TRANSLATIONS: Tuple[Tuple[str, str], ...] = (
        ("司美格鲁肽", "semaglutide"),
        ("临床指南", "clinical guideline"),
        ("指南", "guideline"),
        ("剂量调整", "dose adjustment"),
        ("适应症", "indications"),
        ("适应证", "indications"),
        ("心血管", "cardiovascular"),
        ("安全性", "safety"),
        ("胃肠道", "gastrointestinal"),
        ("不良反应", "adverse reactions"),
        ("不良事件", "adverse events"),
        ("临床证据", "clinical evidence"),
        ("肾功能不全", "renal impairment"),
        ("肾功能", "renal function"),
        ("老年患者", "older adults elderly patients"),
        ("老年", "elderly"),
        ("监测", "monitoring"),
        ("建议", "recommendations"),
        ("糖尿病", "diabetes"),
        ("肥胖", "obesity"),
        ("治疗", "treatment"),
        ("药物", "drug medication"),
        ("研究", "study trial"),
        ("最新", "latest"),
        ("更新", "update"),
    )

    def __init__(self, timeout: int = 30, proxy: Optional[str] = None):
        """
        初始化搜索工具

        Args:
            timeout: HTTP 请求超时时间（秒）
            proxy: 代理地址（如 "http://127.0.0.1:7897"），也可通过环境变量 DDGS_PROXY 设置
        """
        self.timeout = timeout
        self.proxy = proxy or os.environ.get("DDGS_PROXY") or self._detect_system_proxy()

        # 医学领域权威网站白名单
        self.medical_domains = [
            "accessdata.fda.gov",
            "fda.gov",
            "ema.europa.eu",
            "pubmed.ncbi.nlm.nih.gov",
            "pmc.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov",
            "diabetesjournals.org",
            "nejm.org",
            "thelancet.com",
            "jamanetwork.com",
            "bmj.com",
            "cochranelibrary.com",
            "nice.org.uk",
            "mayoclinic.org",
            "webmd.com",
            "who.int",
            "cdc.gov",
            "nih.gov",
            "uptodate.com",
            "medscape.com",
            "healthline.com",
            "medicalnewstoday.com"
        ]

    @staticmethod
    def _detect_system_proxy() -> Optional[str]:
        """
        自动检测系统代理。

        primp (DDGS v8 的 HTTP 客户端) 在 proxy=None 时会自动检测系统代理，
        但可能将 HTTP 代理误识别为 SOCKS 代理导致连接失败。
        此方法显式检测代理并确保使用正确的协议前缀（http://）。
        """
        import socket

        # 1. 检查标准代理环境变量
        for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            val = os.environ.get(var)
            if val:
                logger.debug(f"Using proxy from env {var}: {val}")
                return val

        # 2. Windows: 从注册表读取系统代理
        #    注意：即使 ProxyEnable=0，primp 仍可能读取 ProxyServer 并误用 SOCKS 协议，
        #    因此我们主动检测并用正确的 http:// 前缀覆盖。
        if os.name == "nt":
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                                    r"Software\Microsoft\Windows\CurrentVersion\Internet Settings") as key:
                    proxy_server, _ = winreg.QueryValueEx(key, "ProxyServer")
                    if proxy_server:
                        # 检查代理端口是否实际可达
                        host_port = proxy_server.split("://")[-1]  # 去掉可能的协议前缀
                        parts = host_port.split(":")
                        if len(parts) == 2:
                            host, port = parts[0], int(parts[1])
                            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                            s.settimeout(2)
                            try:
                                if s.connect_ex((host, port)) == 0:
                                    proxy_url = f"http://{host_port}" if "://" not in proxy_server else proxy_server
                                    logger.info(f"Detected active system proxy: {proxy_url}")
                                    return proxy_url
                            finally:
                                s.close()
            except (OSError, FileNotFoundError, ValueError):
                pass

        return None

    async def search(
        self,
        query: str,
        max_results: int = 10,
        region: str = "cn-zh",  # 中国区域，更适合中文搜索
        safesearch: str = "on",  # 严格安全搜索
        timelimit: Optional[str] = None,  # 时间限制：'d'(天), 'w'(周), 'm'(月), 'y'(年)
        retry_count: int = 2  # 重试次数
    ) -> List[SearchResult]:
        """
        执行搜索（参考 shanglv 项目的实现）

        Args:
            query: 搜索查询
            max_results: 最大结果数
            region: 地区设置（cn-zh = 中国区域）
            safesearch: 安全搜索级别（on = 严格）
            timelimit: 时间限制
            retry_count: 重试次数

        Returns:
            搜索结果列表
        """
        if not DDGS_AVAILABLE:
            logger.error("DDGS not available, cannot perform web search")
            return SearchResults()

        max_results = max(1, max_results)
        query_candidates = self._build_query_candidates(query)
        regions = self._fallback_regions(region)
        backends = self._fallback_backends()
        search_plan = self._build_search_plan(query_candidates, regions, backends)
        attempt_limit = min(len(search_plan), max(retry_count + 1, 6))
        last_error: Optional[Exception] = None
        collected_results: List[SearchResult] = []
        attempts_made = 0

        for attempt, (candidate_query, candidate_region, backend) in enumerate(
            search_plan[:attempt_limit],
            start=1
        ):
            attempts_made = attempt
            try:
                logger.info(
                    "Web searching "
                    f"(attempt {attempt}/{attempt_limit}): {candidate_query} "
                    f"(region={candidate_region}, backend={backend}, max_results={max_results})"
                )

                # 使用 DDGS 搜索
                search_results = []

                # v9: backend 可指定多个引擎；旧版不支持时会自动降级为无 backend 参数。
                try:
                    ddgs = DDGS(proxy=self.proxy, timeout=self.timeout)
                    raw = self._ddgs_text(
                        ddgs,
                        candidate_query,
                        max_results=max_results * 2,  # 获取更多结果用于过滤
                        safesearch=safesearch,
                        region=candidate_region,
                        timelimit=timelimit,
                        backend=backend,
                    )
                    search_results = list(raw)
                    if search_results:
                        logger.debug(f"DDGS returned {len(search_results)} results")
                except Exception as e:
                    last_error = e
                    logger.debug(
                        "DDGS search failed for "
                        f"query={candidate_query!r}, region={candidate_region}, backend={backend}: {e}"
                    )

                # 处理搜索结果
                results = self._normalize_results(search_results, max_results=max_results)

                if results:
                    collected_results = self._merge_results(collected_results, results)
                    preferred_results = self._prefer_authoritative_results(collected_results)[:max_results]
                    authoritative_count = sum(
                        1 for result in preferred_results
                        if self._is_authoritative_result(result)
                    )
                    required_authoritative = min(max_results, 2)
                    if authoritative_count >= required_authoritative and len(preferred_results) >= max_results:
                        logger.info(
                            f"Found {len(preferred_results)} results for: {query} "
                            f"(matched candidate: {candidate_query}, authoritative={authoritative_count})"
                        )
                        return SearchResults(
                            preferred_results,
                            retry_count=attempt - 1,
                        )

                    logger.debug(
                        f"Collected {len(collected_results)} candidate results for {query}; "
                        "continuing to seek authoritative sources"
                    )
                else:
                    logger.warning(
                        "No results found for candidate: "
                        f"{candidate_query} (region={candidate_region}, backend={backend})"
                    )

            except Exception as e:
                last_error = e
                logger.warning(f"Web search error (attempt {attempt}): {e}")

                if attempt < attempt_limit:
                    # 等待后重试
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))  # 指数退避

        if last_error:
            logger.error(f"Web search exhausted all fallbacks; last error: {last_error}")

        if collected_results:
            preferred_results = self._prefer_authoritative_results(collected_results)[:max_results]
            logger.info(f"Returning {len(preferred_results)} fallback results for: {query}")
            return SearchResults(
                preferred_results,
                retry_count=max(0, attempts_made - 1),
                retry_exhausted=attempts_made >= attempt_limit,
            )

        return SearchResults(
            retry_count=max(0, attempts_made - 1),
            retry_exhausted=attempts_made >= attempt_limit,
        )

    def _ddgs_text(
        self,
        ddgs: Any,
        query: str,
        *,
        max_results: int,
        safesearch: str,
        region: str,
        timelimit: Optional[str],
        backend: str
    ) -> Iterable[Dict[str, Any]]:
        """兼容新版 ddgs 和旧版 duckduckgo_search 的 text 参数差异。"""
        kwargs: Dict[str, Any] = {
            "max_results": max_results,
            "safesearch": safesearch,
            "region": region,
        }
        if timelimit:
            kwargs["timelimit"] = timelimit
        if backend:
            kwargs["backend"] = backend

        try:
            return ddgs.text(query, **kwargs)
        except TypeError as exc:
            if "backend" not in str(exc):
                raise
            kwargs.pop("backend", None)
            return ddgs.text(query, **kwargs)

    def _normalize_results(
        self,
        raw_results: Iterable[Dict[str, Any]],
        *,
        max_results: int
    ) -> List[SearchResult]:
        """将不同搜索后端的字段统一成 SearchResult，并按 URL 去重。"""
        results: List[SearchResult] = []
        seen_urls = set()

        for result in raw_results:
            url = result.get("href") or result.get("url") or ""
            title = result.get("title") or result.get("heading") or ""
            snippet = result.get("body") or result.get("snippet") or result.get("description") or ""
            if not url or not title:
                continue

            normalized_url = url.rstrip("/")
            if normalized_url in seen_urls:
                continue

            seen_urls.add(normalized_url)
            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source="web"
                )
            )

            if len(results) >= max_results:
                break

        return results

    def _merge_results(
        self,
        existing: List[SearchResult],
        new_results: List[SearchResult]
    ) -> List[SearchResult]:
        seen_urls = {result.url.rstrip("/") for result in existing}
        merged = list(existing)

        for result in new_results:
            normalized_url = result.url.rstrip("/")
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            merged.append(result)

        return merged

    def _prefer_authoritative_results(
        self,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        return [
            result for _, result in sorted(
                enumerate(results),
                key=lambda item: (self._authority_score(item[1]), item[0])
            )
        ]

    def _authority_score(self, result: SearchResult) -> int:
        url = result.url.lower()
        if self._is_authoritative_result(result):
            return 0
        if any(domain in url for domain in self.medical_domains):
            return 1
        if ".gov/" in url or ".edu/" in url or ".org/" in url:
            return 2
        return 3

    @staticmethod
    def _is_authoritative_result(result: SearchResult) -> bool:
        authoritative_domains = (
            "accessdata.fda.gov",
            "fda.gov",
            "ema.europa.eu",
            "pubmed.ncbi.nlm.nih.gov",
            "pmc.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov",
            "nih.gov",
            "who.int",
            "cdc.gov",
            "nice.org.uk",
            "diabetesjournals.org",
            "nejm.org",
            "thelancet.com",
            "jamanetwork.com",
            "bmj.com",
            "cochranelibrary.com",
        )
        url = result.url.lower()
        return any(domain in url for domain in authoritative_domains)

    def _build_query_candidates(self, query: str) -> List[str]:
        """为中文医学长查询生成更容易被通用搜索引擎命中的候选查询。"""
        query = query.strip()
        if not query:
            return []

        candidates = [query]
        if "医学" not in query and "medical" not in query.lower():
            candidates.append(f"{query} 医学 临床 指南 证据")

        english_query = self._translate_medical_query(query)
        if english_query:
            candidates.extend([
                english_query,
                f"{english_query} guideline clinical evidence",
                f"{english_query} FDA label EMA product information",
                f"{english_query} PubMed clinical trial review",
            ])

        return self._dedupe_strings(candidates)

    def _translate_medical_query(self, query: str) -> str:
        """用轻量词表抽取英文医学检索词，避免长中文句子直接搜不到结果。"""
        translated_terms: List[str] = []
        remaining = query

        for chinese, english in self.MEDICAL_TRANSLATIONS:
            if chinese in remaining:
                translated_terms.extend(english.split())
                remaining = remaining.replace(chinese, " ")

        latin_terms = re.findall(r"[A-Za-z][A-Za-z0-9-]*|\d{4}", remaining)
        terms = self._dedupe_strings(latin_terms + translated_terms)

        if len(terms) < 2:
            return ""

        return " ".join(terms)

    @staticmethod
    def _dedupe_strings(values: Iterable[str]) -> List[str]:
        """Deduplicate strings by case-insensitive normalized form (public utility)."""
        seen = set()
        deduped = []
        for value in values:
            normalized = " ".join(value.split())
            key = normalized.lower()
            if not normalized or key in seen:
                continue
            seen.add(key)
            deduped.append(normalized)
        return deduped

    @staticmethod
    def dedupe_results(results: List[SearchResult]) -> List[SearchResult]:
        """Deduplicate search results by URL (public utility method)."""
        deduped: List[SearchResult] = []
        seen_urls: set = set()
        for result in results:
            normalized_url = result.url.rstrip("/")
            if not normalized_url or normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            deduped.append(result)
        return deduped

    @staticmethod
    def _fallback_regions(region: str) -> List[str]:
        regions = [region, "us-en", "wt-wt"]
        return WebSearchTool._dedupe_strings(regions)

    @staticmethod
    def _fallback_backends() -> List[str]:
        return [
            "auto",
            "duckduckgo,bing,brave",
            "google,yahoo,startpage",
        ]

    @staticmethod
    def _build_search_plan_lazy(
        queries: List[str],
        regions: List[str],
        backends: List[str]
    ) -> Iterable[Tuple[str, str, str]]:
        """Lazily generate search plan combinations instead of pre-building full list."""
        # First: preferred query with each region/backend
        for candidate_query in queries:
            yield (candidate_query, regions[0], backends[0])
        # Then: remaining combinations
        for candidate_query in queries:
            for candidate_region in regions:
                for backend in backends:
                    item = (candidate_query, candidate_region, backend)
                    if (candidate_query, regions[0], backends[0]) != item:
                        yield item

    @staticmethod
    def _build_search_plan(
        queries: List[str],
        regions: List[str],
        backends: List[str]
    ) -> List[Tuple[str, str, str]]:
        return list(WebSearchTool._build_search_plan_lazy(queries, regions, backends))

    def filter_by_domain(
        self,
        results: List[SearchResult],
        allowed_domains: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        按域名过滤结果

        Args:
            results: 搜索结果
            allowed_domains: 允许的域名列表（默认使用医学域名白名单）

        Returns:
            过滤后的结果
        """
        if allowed_domains is None:
            allowed_domains = self.medical_domains

        filtered = []
        for result in results:
            # 检查 URL 是否包含白名单域名
            if any(domain in result.url for domain in allowed_domains):
                filtered.append(result)

        logger.info(f"Filtered {len(filtered)}/{len(results)} results by domain")
        return filtered

    async def fetch_content(
        self,
        url: str,
        max_length: int = 2000
    ) -> Optional[str]:
        """
        抓取网页内容

        Args:
            url: 网页 URL
            max_length: 最大内容长度

        Returns:
            网页文本内容（提取正文）
        """
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url)
                response.raise_for_status()

                # 使用 BeautifulSoup 提取正文
                soup = BeautifulSoup(response.text, 'html.parser')

                # 移除 script 和 style 标签
                for script in soup(["script", "style"]):
                    script.decompose()

                # 提取文本
                text = soup.get_text()

                # 清理空白字符
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)

                # 限制长度
                if len(text) > max_length:
                    text = text[:max_length] + "..."

                return text

        except Exception as e:
            logger.error(f"Failed to fetch content from {url}: {e}")
            return None

    async def search_with_content(
        self,
        query: str,
        max_results: int = 5,
        fetch_full_content: bool = False
    ) -> List[Dict[str, Any]]:
        """
        搜索并获取内容

        Args:
            query: 搜索查询
            max_results: 最大结果数
            fetch_full_content: 是否抓取完整内容

        Returns:
            包含内容的搜索结果
        """
        # 执行搜索
        results = await self.search(query, max_results=max_results)

        # 如果需要，抓取完整内容
        enriched_results = []
        for result in results:
            enriched = {
                "title": result.title,
                "url": result.url,
                "snippet": result.snippet,
                "source": result.source,
                "full_content": None
            }

            if fetch_full_content:
                content = await self.fetch_content(result.url)
                enriched["full_content"] = content

            enriched_results.append(enriched)

        return enriched_results


# 便捷函数
async def search_medical_web(
    query: str,
    max_results: int = 10
) -> List[SearchResult]:
    """
    快速搜索医学网络信息

    Args:
        query: 搜索查询
        max_results: 最大结果数

    Returns:
        搜索结果列表
    """
    tool = WebSearchTool()
    return await tool.search(query, max_results=max_results)
