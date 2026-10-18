# Complete scheduler report

**The Picture Messenger** is a 64-page, plain-language explanation of the project, with about 17,000 words across 31 chapters.

- [Read the report on GitHub](complete-report.md)
- [Download the PDF](complete-report.pdf)
- [Download the standalone HTML](complete-report.html)
- [Download the complete report and evidence bundle](semantic-scheduler-report.zip)

The report explains the implemented scheduling policies, the six-job demonstration, semantic measurements, knowledge bases, codebooks, image codecs, proposed static and dynamic methods, benchmark design, related research, and production requirements. Worked examples, diagrams and a glossary introduce the concepts in simple language.

It distinguishes **existing implementation**, **checks performed for the report**, **earlier recorded evidence**, and **proposed future work**. The proposed semantic/dynamic policies are not presented as implemented or validated capabilities.

The reviewed application revision is [`edad7d07c44fd3006e4359b8c32b6dbe143818d7`](https://github.com/saltypal/SemanticSchedulerEnd2End/tree/edad7d07c44fd3006e4359b8c32b6dbe143818d7). Selected tests run for the report yielded **25 passed, 1 skipped and 2 external-worker failures**, explained in the evidence ledger. This is not a full-suite result, and no new trained-model inference or neural training was performed.

Supporting [planning documents](planning/), [execution evidence](execution/), [figures](assets/), and [report-generation scripts](tools/) are included. The reading copies work without rerunning the scripts; generation scripts retain the original workspace paths. HTML can be downloaded and opened locally; uploading it here does not publish a GitHub Pages site.
