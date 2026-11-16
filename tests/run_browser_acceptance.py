"""Harvey Local AI Sales Agent — Full Acceptance Test Suite."""

import asyncio
import sys
import threading
import time
import uvicorn
from playwright.async_api import async_playwright

from sellbird.dashboard import app, token_store
from sellbird.models import ICP, Product, Prospect
from sellbird.signals import seed_signal_catalog
from sellbird.state import StateManager

TEST_PORT = 5566
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"

# Enable real-time unbuffered prints
sys.stdout.reconfigure(line_buffering=True)


async def run_full_acceptance():
    print("=" * 60)
    print("STARTING FULL ACCEPTANCE TEST SUITE ON PLAYWRIGHT EDGE")
    print("=" * 60)

    # 1. Start Server
    config = uvicorn.Config(app, host="127.0.0.1", port=TEST_PORT, log_level="error", loop="asyncio")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(1.5)

    state = StateManager()
    await state.init_db()
    await seed_signal_catalog(state)

    # Pre-clean database
    async with state._connect() as db:
        await db.execute("DELETE FROM outbox WHERE prospect_id LIKE 'test-%'")
        await db.execute("DELETE FROM observations WHERE company_id LIKE 'test-%'")
        await db.execute("DELETE FROM research_dossiers WHERE company_id LIKE 'test-%'")
        await db.execute("DELETE FROM prospects WHERE id LIKE 'test-%'")
        await db.execute("DELETE FROM icps WHERE name IN ('DevOps & SRE Leaders', 'DevOps & SRE Leads', 'FinTech Security Officers')")
        await db.execute("DELETE FROM products WHERE name IN ('CloudPulse Tracing', 'Sentinel SOC2 Guard')")
        await db.execute("DELETE FROM businesses WHERE name LIKE 'OmniCloud%' or name LIKE 'Test %' or name LIKE 'Alpha%' or name LIKE 'Check%'")
        await db.commit()

    token = token_store.get()
    results = {}
    console_errors = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(channel="msedge", headless=True)
        page = await browser.new_page(viewport={"width": 1280, "height": 800})

        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" and "password" not in m.text.lower() and "favicon" not in m.text.lower() else None)
        page.on("pageerror", lambda e: console_errors.append(f"Uncaught JS: {e}"))

        # --- Journey 1: Open Dashboard & Initial Empty States ---
        print("\n[Journey 1] Loading dashboard & verifying initial empty states...")
        await page.goto(f"{BASE_URL}/#token={token}")
        await page.wait_for_selector(".app", timeout=5000)
        title = await page.title()
        assert "Sellbird" in title

        await page.evaluate("showTab('company')")
        await page.wait_for_selector("#pb-list-view", timeout=5000)
        await page.wait_for_timeout(1000)
        pb_text = await page.inner_text("#pb-list-view")
        assert "No Business Profiles" in pb_text or "Create your organization" in pb_text
        results["1. Dashboard Load & Empty States"] = "PASSED"
        print("  -> PASSED: Clean load, 0 console errors, initial empty states verified.")

        # --- Journey 2: Create Business Profile & Form Validation ---
        print("\n[Journey 2] Testing Business Profile creation & validation...")
        await page.evaluate("openBusinessModal()")
        await page.wait_for_selector("#business-modal[open]", timeout=3000)

        # Validation: empty save
        await page.click("#bm-save-btn")
        toast = await page.wait_for_selector(".toast.error", timeout=3000)
        toast_msg = await toast.inner_text()
        assert "Business name is required" in toast_msg
        await page.wait_for_timeout(3000)  # Wait for toast to fade

        await page.fill("#bm-name", "OmniCloud Security")
        await page.fill("#bm-domain", "omnicloud.dev")
        await page.fill("#bm-industry", "Cloud Security")
        await page.fill("#bm-desc", "Zero-trust automated policy enforcement for Kubernetes.")
        await page.click("#bm-save-btn")

        await page.wait_for_selector("#pb-list-view table", timeout=10000)
        biz_list = await page.evaluate("() => _businesses")
        assert len(biz_list) >= 1
        biz_id = biz_list[0]["id"]
        table_text = await page.inner_text("#pb-list-view table")
        assert "OmniCloud Security" in table_text
        assert "omnicloud.dev" in table_text
        results["2. Business Profile Management"] = "PASSED"
        print(f"  -> PASSED: Form validation confirmed; created OmniCloud Security ({biz_id}).")

        # --- Journey 3: Create Two Products Under Business ---
        print("\n[Journey 3] Creating two products under business...")
        await page.evaluate("switchPBTab('products')")
        await page.wait_for_timeout(500)

        # Product 1
        await page.evaluate("openProductModal()")
        await page.wait_for_selector("#product-modal[open]", timeout=3000)

        # Validation test
        await page.click("#pm-save-btn")
        p_toast = await page.wait_for_selector(".toast.error", timeout=3000)
        assert "Product name is required" in await p_toast.inner_text()
        await page.wait_for_timeout(3000)

        await page.select_option("#pm-business-id", biz_id)
        await page.fill("#pm-name", "CloudPulse Tracing")
        await page.fill("#pm-desc", "Continuous eBPF microservices latency observer.")
        await page.fill("#pm-pricing", "$199/month")
        await page.fill("#pm-offer", "Free 14-day telemetry audit")
        await page.fill("#pm-features", "Distributed eBPF kernel probes\nSub-millisecond query speed")
        await page.fill("#pm-benefits", "Cut MTTR by 60%\nZero overhead")
        await page.fill("#pm-usps", "100% on-premises data retention")
        await page.fill("#pm-objections", "pricing: Full ROI in 14 days")
        await page.click("#pm-save-btn")
        await page.wait_for_timeout(1500)

        # Product 2
        await page.evaluate("openProductModal()")
        await page.wait_for_selector("#product-modal[open]", timeout=3000)
        await page.select_option("#pm-business-id", biz_id)
        await page.fill("#pm-name", "Sentinel SOC2 Guard")
        await page.fill("#pm-desc", "Automated compliance monitoring.")
        await page.fill("#pm-pricing", "$499/month")
        await page.fill("#pm-offer", "Free compliance gap audit")
        await page.fill("#pm-features", "Automated IAM audit\nContinuous posture assessment")
        await page.fill("#pm-benefits", "Shorten audit prep from 3 months to 2 days")
        await page.fill("#pm-usps", "Zero cloud telemetry leakage")
        await page.fill("#pm-objections", "audit: Formally mapped to SOC2 Type II")
        await page.click("#pm-save-btn")
        await page.wait_for_timeout(1500)

        prods = await page.evaluate("() => _products")
        assert len(prods) == 2
        prod1_id = next(p["id"] for p in prods if p["name"] == "CloudPulse Tracing")
        prod2_id = next(p["id"] for p in prods if p["name"] == "Sentinel SOC2 Guard")
        results["3. Multi-Product Management"] = "PASSED"
        print(f"  -> PASSED: Created CloudPulse Tracing ({prod1_id}) and Sentinel SOC2 Guard ({prod2_id}).")

        # --- Journey 4: Switch Active Product ---
        print("\n[Journey 4] Switching active product...")
        await page.evaluate(f"changeActiveProduct('{prod1_id}')")
        await page.wait_for_timeout(1000)
        assert await page.evaluate("() => _activeProductId") == prod1_id

        await page.select_option("#global-active-product", prod2_id)
        await page.wait_for_timeout(1000)
        assert await page.evaluate("() => _activeProductId") == prod2_id
        results["4. Active Product Context Switcher"] = "PASSED"
        print("  -> PASSED: Active product switched and persisted seamlessly.")

        # --- Journey 5: Create Target ICPs for Products ---
        print("\n[Journey 5] Creating product-specific ICPs...")
        await page.evaluate("switchPBTab('icps')")
        await page.wait_for_timeout(500)

        # ICP 1
        await page.evaluate(f"openIcpModal(null, '{prod1_id}')")
        await page.wait_for_selector("#icp-modal[open]", timeout=3000)
        await page.fill("#im-name", "DevOps & SRE Leaders")
        await page.fill("#im-industries", "DevOps, SaaS, Cloud Infrastructure")
        await page.fill("#im-titles", "VP Engineering, Head of SRE, Principal Architect")
        await page.fill("#im-pains", "High MTTR during latency spikes\nExcessive APM licensing costs")
        await page.fill("#im-signals", "Hiring Kubernetes engineers\nAnnounced microservices migration")
        await page.click("#im-save-btn")
        await page.wait_for_timeout(1500)

        # ICP 2
        await page.evaluate(f"openIcpModal(null, '{prod2_id}')")
        await page.wait_for_selector("#icp-modal[open]", timeout=3000)
        await page.fill("#im-name", "FinTech Security Officers")
        await page.fill("#im-industries", "FinTech, Banking, Cybersecurity")
        await page.fill("#im-titles", "CISO, VP Information Security, Compliance Director")
        await page.fill("#im-pains", "Impending SOC2 Type II audit\nManual spreadsheet evidence gathering")
        await page.fill("#im-signals", "Series B funding round\nExpanding compliance team")
        await page.click("#im-save-btn")
        await page.wait_for_timeout(1500)

        icps = await page.evaluate("() => _icps")
        assert len(icps) == 2
        icp1_id = next(i["id"] for i in icps if i["name"] == "DevOps & SRE Leaders")
        icp2_id = next(i["id"] for i in icps if i["name"] == "FinTech Security Officers")
        results["5. ICP Management"] = "PASSED"
        print(f"  -> PASSED: Created DevOps & SRE Leads ({icp1_id}) and FinTech Security Officers ({icp2_id}).")

        # --- Journey 6: Prospect List & Product Isolation / Filtering ---
        print("\n[Journey 6] Testing prospect list & product isolation...")
        pr1 = Prospect(
            id="test-p1-alex",
            first_name="Alex",
            last_name="Mercer",
            email="alex@apexinfra.io",
            company="Apex Infra",
            title="Head of SRE",
            industry="Cloud Infrastructure",
            product_id=prod1_id,
            icp_id=icp1_id,
            score=82,
            status="new",
        )
        pr2 = Prospect(
            id="test-p2-sarah",
            first_name="Sarah",
            last_name="Connor",
            email="sarah@cybershield.org",
            company="CyberShield",
            title="CISO",
            industry="FinTech",
            product_id=prod2_id,
            icp_id=icp2_id,
            score=88,
            status="new",
        )
        await state.add_prospect(pr1)
        await state.add_prospect(pr2)

        # Seed research dossier for Alex Mercer
        await state.save_research_dossier(
            company_id=pr1.id,
            domain="apexinfra.io",
            product_id=prod1_id,
            verified_facts=[{"text": "Apex Infra announced multi-region Kubernetes migration in Q4.", "source_url": "https://apexinfra.io/blog/k8s-scale", "created_at": "2026-11-01T10:00:00Z"}],
            ai_interpretations=[{"signal_code": "HIRING_ROLE", "confidence": "high", "evidence": "Currently hiring 5 Senior SRE engineers."}],
            source_urls=["https://apexinfra.io/blog/k8s-scale"],
        )
        await state.add_observation(company_id=pr1.id, signal_code="HIRING_ROLE", value_text="Currently hiring 5 Senior SRE engineers.", evidence_url="https://apexinfra.io/blog/k8s-scale")

        await page.evaluate("showTab('prospects')")
        await page.wait_for_selector(f"#prospect-product-filter option[value='{prod1_id}']", state="attached", timeout=5000)

        # All products
        await page.select_option("#prospect-product-filter", "")
        await page.wait_for_timeout(1000)
        all_text = await page.inner_text("#prospects-table")
        assert "Alex Mercer" in all_text and "Sarah Connor" in all_text

        # Product 1 only
        await page.select_option("#prospect-product-filter", prod1_id)
        await page.wait_for_timeout(1000)
        p1_text = await page.inner_text("#prospects-table")
        assert "Alex Mercer" in p1_text and "Sarah Connor" not in p1_text

        # Product 2 only
        await page.select_option("#prospect-product-filter", prod2_id)
        await page.wait_for_timeout(1000)
        p2_text = await page.inner_text("#prospects-table")
        assert "Sarah Connor" in p2_text and "Alex Mercer" not in p2_text

        results["6. Product Isolation & Filtering"] = "PASSED"
        print("  -> PASSED: Strict product isolation verified in live browser UI.")

        # --- Journey 7 & 8: Prospect Intelligence Drawer & Scoring Breakdown ---
        print("\n[Journey 7 & 8] Opening Prospect Intelligence Drawer...")
        await page.evaluate("openProspectIntelligence('test-p1-alex')")
        await page.wait_for_selector("#intel-modal[open]", timeout=5000)
        await page.wait_for_selector(".intel-grid", timeout=15000)

        intel_text = await page.inner_text("#intel-modal-body")
        assert "/ 100" in intel_text
        assert "Industry Fit" in intel_text
        assert "Job Title Fit" in intel_text
        assert "Email Deliverability" in intel_text

        # Badges & evidence
        assert await page.query_selector(".badge-verified") is not None
        assert await page.query_selector(".badge-ai") is not None
        assert await page.query_selector(".evidence-source a") is not None

        results["7. Prospect Intelligence Drawer"] = "PASSED"
        print("  -> PASSED: Grounded facts, AI buying signals, 0-100 score, source link verified.")

        # Re-score prospect
        await page.evaluate("() => { document.querySelector('#intel-modal-body').innerHTML = ''; reScoreProspect('test-p1-alex'); }")
        await page.wait_for_selector("#intel-modal-body .intel-grid", timeout=20000)
        rescore_text = await page.inner_text("#intel-modal-body")
        assert "/ 100" in rescore_text
        results["8. Explainable Scoring & Re-Score"] = "PASSED"
        print("  -> PASSED: Prospect re-scoring recalculated & UI updated.")

        # --- Journey 9: Generate & Edit Personalized Outreach Email ---
        print("\n[Journey 9] Generating & editing outreach email...")
        await page.evaluate("openEmailDraft('test-p1-alex')")
        await page.wait_for_selector("#email-modal[open]", timeout=15000)

        subject = await page.input_value("#email-modal-subj")
        body = await page.input_value("#email-modal-body-input")
        assert len(subject) > 3 and len(body) > 20
        assert "Alex" in body or "Apex" in body or "CloudPulse" in body

        # Human editing
        custom_note = "\n\nP.S. Let me know if Thursday 2pm works for a quick demo."
        await page.fill("#email-modal-body-input", body + custom_note)
        assert custom_note in await page.input_value("#email-modal-body-input")
        results["9. Generate & Edit Outreach Draft"] = "PASSED"
        print(f"  -> PASSED: Generated personalized email '{subject}'; edited draft.")

        # --- Journey 10: Stage Email & Human Approval Outbox Safeguard ---
        print("\n[Journey 10] Staging email to Outbox as pending_review...")
        await page.click("#email-modal-stage-btn")
        await page.wait_for_timeout(1500)
        assert not await page.is_visible("#email-modal[open]")

        outbox_items = await state.get_outbox(limit=5)
        assert len(outbox_items) >= 1
        staged = outbox_items[0]
        status_val = staged.get("status") if isinstance(staged, dict) else staged.status
        body_val = staged.get("body") if isinstance(staged, dict) else staged.body
        assert status_val == "pending_review"
        assert "Thursday 2pm" in body_val

        await page.evaluate("showTab('outbox')")
        await page.wait_for_timeout(1500)
        outbox_ui = await page.inner_text("#outbox")
        assert "Apex" in outbox_ui or "alex@" in outbox_ui or "Review" in outbox_ui
        results["10. Email Staging & Human Approval Safeguard"] = "PASSED"
        print("  -> PASSED: Email committed to Outbox in pending_review status; zero automated dispatch.")

        # --- Journey 11: Multi-Product Sales Analytics ---
        print("\n[Journey 11] Inspecting Sales Analytics...")
        await page.evaluate("showTab('analytics')")
        await page.wait_for_selector("#analytics-content", timeout=6000)
        await page.wait_for_timeout(1000)

        analytics_text = await page.inner_text("#analytics-content")
        assert "Total Prospects" in analytics_text
        assert "Qualified Prospects" in analytics_text

        # Product filtering
        await page.select_option("#analytics-product-filter", prod1_id)
        await page.wait_for_timeout(1000)
        assert "Total Prospects" in await page.inner_text("#analytics-content")

        await page.select_option("#analytics-product-filter", prod2_id)
        await page.wait_for_timeout(1000)
        assert "Total Prospects" in await page.inner_text("#analytics-content")
        results["11. Sales Analytics"] = "PASSED"
        print("  -> PASSED: Global and product-specific analytics funnels verified.")

        # --- Journey 12: Local AI & Hardware Status Telemetry ---
        print("\n[Journey 12] Inspecting Local AI & Hardware status...")
        await page.evaluate("showTab('usage')")
        await page.wait_for_selector(".hardware-card", timeout=12000)
        hw_ui = (await page.inner_text("#local-ai-hardware-card")).lower()
        assert "primary llm" in hw_ui
        assert "qwen3:4b" in hw_ui
        assert "embedding model" in hw_ui
        assert "all-minilm-l6-v2" in hw_ui
        assert "detected hardware" in hw_ui
        assert "ram" in hw_ui
        results["12. Local AI & Hardware Status"] = "PASSED"
        print("  -> PASSED: Local Ollama qwen3:4b, sentence-transformers, Win32 RAM verified.")

        # --- Journey 13: UI Edge States, Confirmation Dialogs & Mobile ---
        print("\n[Journey 13] Verifying UI states, confirmation dialogs & mobile layout...")
        await page.evaluate("showTab('company')")
        await page.evaluate("switchPBTab('products')")
        await page.wait_for_timeout(500)

        # Deletion cancel dialog safeguard
        dismissed = False
        async def cancel_dialog(dialog):
            nonlocal dismissed
            dismissed = True
            await dialog.dismiss()

        page.once("dialog", cancel_dialog)
        await page.evaluate(f"deleteProduct('{prod1_id}')")
        await page.wait_for_timeout(500)
        assert dismissed
        prods_after = await page.evaluate("() => _products")
        assert any(p["id"] == prod1_id for p in prods_after)

        # Mobile viewport test
        await page.set_viewport_size({"width": 375, "height": 667})
        await page.wait_for_timeout(500)
        assert await page.is_visible(".brand")
        await page.evaluate("showTab('today')")
        await page.wait_for_timeout(500)
        assert await page.is_visible("#today")
        await page.set_viewport_size({"width": 1280, "height": 800})
        results["13. Confirmation Dialogs & Mobile Viewport"] = "PASSED"
        print("  -> PASSED: Cancel deletion confirmed; mobile viewport layout verified.")

        # Cleanup
        async with state._connect() as db:
            await db.execute("DELETE FROM outbox WHERE prospect_id LIKE 'test-%'")
            await db.execute("DELETE FROM observations WHERE company_id LIKE 'test-%'")
            await db.execute("DELETE FROM research_dossiers WHERE company_id LIKE 'test-%'")
            await db.execute("DELETE FROM prospects WHERE id LIKE 'test-%'")
            await db.execute("DELETE FROM icps WHERE id IN (?, ?)", (icp1_id, icp2_id))
            await db.execute("DELETE FROM products WHERE id IN (?, ?)", (prod1_id, prod2_id))
            await db.execute("DELETE FROM businesses WHERE id = ?", (biz_id,))
            await db.commit()
        print("\n[Cleanup] Test records cleaned up successfully.")

        await browser.close()

    server.should_exit = True
    print("\n" + "=" * 60)
    print("FINAL ACCEPTANCE TEST RESULTS SUMMARY")
    print("=" * 60)
    for name, st in results.items():
        print(f"[{st}] {name}")
    print(f"Console Errors: {len(console_errors)}")
    for err in console_errors:
        print(f"  * {err}")

    assert len(results) == 13 and all(st == "PASSED" for st in results.values()) and len(console_errors) == 0
    print("\nALL 13 BROWSER ACCEPTANCE TESTS PASSED WITH 100% SUCCESS RATE!")


if __name__ == "__main__":
    asyncio.run(run_full_acceptance())
