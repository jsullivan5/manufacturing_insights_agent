# AVEVA PI System Time Series Data Format and Schema for Mock Dataset Generation

The AVEVA PI System, formerly known as OSIsoft PI System, utilizes specific data structures and formats for time series data ingestion and export, particularly when working with CSV files for demo projects and data exchange. Understanding these formats is crucial for creating realistic mock datasets that accurately simulate industrial time series data from systems like freezer monitoring applications.

## Standard CSV Export Formats in PI System

### Long Format Structure (Tag-Centric)

The PI System typically employs a **long format** (also called "narrow" format) for CSV exports, where each row represents a single timestamp-value pair for a specific tag[5][9]. This approach contrasts with wide formats where multiple tags would be represented as columns across a single timestamp row. The long format is preferred because it accommodates the irregular sampling intervals common in industrial data collection, where different sensors may record data at different frequencies.

The standard field structure for PI System CSV exports includes several core components that ensure data integrity and traceability. The timestamp field serves as the primary temporal reference, typically formatted in a standardized date-time format that preserves millisecond precision when necessary[5]. The tag name field provides the unique identifier for each data point, following PI System naming conventions that often incorporate hierarchical structures reflecting the physical or logical organization of the monitored equipment.

### Core Field Components

The typical PI System CSV schema encompasses **timestamp**, **tag name**, **value**, and **quality/attributes** fields[5]. The timestamp field maintains chronological ordering and supports various time zone specifications, which is critical for industrial systems operating across multiple geographic locations. Tag names in PI System follow specific naming conventions that often include descriptive hierarchies, such as "FREEZER01.COMPRESSOR.POWER" or "COLD_STORAGE.TEMP.INTERNAL", which facilitate intuitive data organization and retrieval.

Value fields accommodate multiple data types including numeric measurements, string-based status indicators, and Boolean states for digital signals like door status or alarm conditions[5]. The quality or attributes field provides metadata about the data point's reliability, indicating whether the value was measured, calculated, interpolated, or if any issues occurred during data collection. Common quality indicators include "Good," "Bad," "Questionable," and specific codes for different types of data collection anomalies.

### Multi-Tag Export Considerations

Multi-tag exports in PI System can be structured in different ways depending on the intended use case and the tools being utilized[9]. The most common approach involves creating separate CSV files for each tag, particularly when extracting large volumes of historical data[9]. This method ensures optimal file management and processing efficiency, especially when dealing with tags that have different sampling rates or data types.

However, PI System also supports consolidated multi-tag exports where multiple tags are included in a single CSV file using the long format structure[5]. In this arrangement, each row contains the tag identifier, timestamp, value, and quality information for a single measurement point. This format proves particularly useful for datasets where temporal correlation between different measurements is important, such as in your freezer system example where compressor power, internal temperature, and door status readings need to be analyzed together.

## Practical Implementation for Freezer System Demo

### Recommended Schema Structure

For your freezer system mock dataset, the optimal CSV structure should include columns for **Timestamp**, **TagName**, **Value**, **Units**, and **Quality**[5]. The timestamp should use ISO 8601 format or PI System's standard datetime representation to ensure compatibility with PI System tools. Tag names should follow a hierarchical naming convention such as "FREEZER_SYSTEM.COMPRESSOR.POWER_KW", "FREEZER_SYSTEM.TEMPERATURE.INTERNAL_C", and "FREEZER_SYSTEM.DOOR.STATUS" to reflect the physical system organization.

The units field provides essential context for numeric values, particularly important for engineering applications where measurement units must be explicitly specified[5]. For your freezer system, this would include units like "kW" for compressor power, "°C" for temperature measurements, and "Boolean" or "On/Off" for door status indicators. The quality field should predominantly contain "Good" values for normal operation, with occasional "Bad" or "Questionable" entries to simulate realistic data collection scenarios including sensor malfunctions or communication interruptions.

### Data Generation Considerations

When generating realistic mock data using Python, consider implementing temporal patterns that reflect actual freezer operation cycles. Compressor power should exhibit cyclical behavior corresponding to cooling cycles, with power consumption varying based on ambient conditions and door opening events. Internal temperature should show inverse correlation with compressor operation, gradually rising during off-cycles and decreasing during active cooling phases.

Door status events should be implemented as discrete state changes with realistic timing patterns, such as brief opening periods during business hours for a commercial freezer system[6]. The temporal relationships between these variables should reflect real-world physics, where door opening events cause temperature spikes that trigger compressor activation after appropriate delay periods.

## Integration with PI System Tools

### Import Capabilities and Compatibility

The CSV format you generate should be compatible with PI System's data import tools, particularly the PI Connector for UFL (Universal File Loader) which can parse structured text files and CSV formats for data ingestion[6]. This connector supports various timestamp formats and can handle the long format structure recommended for your mock dataset. Ensuring compatibility with these tools will make your demo dataset more realistic and potentially usable for actual PI System demonstrations.

The PI Tag Configurator add-in for Excel provides additional capabilities for managing tag configurations and can import tag definitions from CSV files[8]. This tool proves valuable when setting up large numbers of tags for demonstration purposes, allowing you to define tag properties, engineering units, and other metadata through spreadsheet interfaces before importing into the PI System.

## Conclusion

The AVEVA PI System's CSV format typically employs a long format structure with timestamp, tag name, value, and quality fields as core components. For your freezer system mock dataset, implementing this structure with appropriate hierarchical tag naming, realistic temporal patterns, and proper quality indicators will create a convincing demonstration environment. The format's flexibility accommodates both single-tag and multi-tag export scenarios, with the long format providing optimal compatibility with PI System tools and realistic representation of industrial time series data collection patterns. This approach will ensure your Python-generated mock data accurately simulates real PI System data structures while providing the temporal complexity necessary for meaningful freezer system demonstrations.

Citations:
[1] https://pisquare.osisoft.com/s/question/0D51I00004T3jOSSAZ/getting-data-out-of-pi-to-a-csv
[2] https://pisquare.osisoft.com/s/topic/0TO1I000000OPxsWAG/data-export
[3] https://docs.aveva.com/bundle/pi-server-l-da-smt/page/1019058.html
[4] https://docs.aveva.com/bundle/pi-server-s-da-reference/page/1221461.html
[5] https://www.youtube.com/watch?v=ULd3n2NFkn4
[6] https://www.youtube.com/watch?v=Xe0631CxRVU
[7] https://forums.raspberrypi.com/viewtopic.php?t=177852
[8] https://www.youtube.com/watch?v=vg7oYwZFNZo
[9] https://github.com/cloud-rocket/PiTagsExtract
[10] http://cdn.osisoft.com/learningcontent/pdfs/PISystemArchitecturePlanningAndImplementationWorkbook.pdf
[11] https://stackoverflow.com/questions/54542341/how-to-export-table-schema-into-a-csv-file
[12] https://stackoverflow.com/questions/53380103/how-do-i-increase-the-default-column-width-of-a-csv-file-so-that-when-i-open-the
[13] https://visplore.com/documentation/v2024a/dataimport/pims.html
[14] https://stackoverflow.com/questions/12259562/how-can-i-create-schema-ini-file-i-need-to-export-my-csv-file-to-datagridview
[15] https://www.youtube.com/watch?v=wFXAXEadgzM
[16] https://pisquare.osisoft.com/0D51I00004UHlWbSAL
[17] https://pisquare.osisoft.com/s/question/0D51I00004UHjjZSAT/what-types-of-files-can-a-pi-ufl-interface-read-and-process
[18] https://osisoft.my.site.com/PISquare/s/topic/0TO1I000000OG9ZWAW/pi-interface-for-universal-file-and-stream-loading-ufl
[19] https://github.com/osisoft/PI-Connector-for-UFL-Samples/blob/master/INI_FILE_EXAMPLES/Example4_StructuredMatrix2.csv
[20] https://www.youtube.com/watch?v=Wdd4tNDofFs
[21] https://www.youtube.com/watch?v=5XkbXniS5pE
[22] https://osicdn.blob.core.windows.net/learningcontent/pdfs/2019%20PI%20World%20Programmability%20in%20the%20PI%20Connector%20for%20UFL.pdf
[23] https://docs.n3uron.com/docs/ufl-exporter-introduction
[24] https://www.cse.lehigh.edu/~brian/PIW18/PIWorld-IntroTimeBasedData.pdf
[25] https://www.reddit.com/r/PLC/comments/19eid3f/pi_datalink_function_to_return_a_timestamp/
[26] https://osicdn.blob.core.windows.net/learningcontent/Online%20Course%20Workbooks/PI%20Vision%20Basics-v3.pdf
[27] https://stackoverflow.com/questions/77569925/how-to-convert-the-timestamp-exported-in-csv-form-to-string-type-in-apache-iotdb
[28] https://github.com/Hugovdberg/PIconnect/issues/526
[29] https://pisquare.osisoft.com/s/Blog-Detail/a8r1I000000GvimQAC/osisoftpowershell-script-to-output-a-bunch-of-csv-files-for-a-list-of-pi-points
[30] https://osisoft.my.site.com/PISquare/s/question/0D51I00004UHdcxSAD/sample-data
[31] https://docs.predictiveindex.com/en/articles/10579976-exporting-organization-data
[32] https://www.plctalk.net/threads/direct-extract-from-osi-pi-ft-historian.135219/
[33] https://support.cosworth.com/post/pi-export-to-excel-11102030
[34] https://pisquare.osisoft.com/s/question/0D51I00004UHghxSAD/exporting-data-from-pi-process-book
[35] https://stackoverflow.com/questions/58867643/read-csv-data-with-ace-oledb-engine-how-to-detect-errors-in-data
[36] https://stackoverflow.com/questions/57348248/python-web-api-to-csv
[37] https://www.youtube.com/watch?v=jfu8xkuLrd0
[38] https://github.com/sergejhorvat/PIwebAPI4R
[39] https://pisquare.osisoft.com/s/question/0D58W00009txLTeSAM/how-to-import-csv-file-to-pi-server-without-pi-connector-for-ufl
[40] https://docs.aveva.com/bundle/pi-web-api-reference/page/help.html
[41] https://osisoft.my.site.com/PISquare/s/question/0D58W00007yYOYhSAO/is-there-any-api-for-extracting-raw-data-from-pi-system-using-pi-integrator
[42] https://github.com/AVEVA/sample-pi_web_api-data_analysis_jupyter-python
[43] https://documentation.trendminer.com/en/84190-configure-pi-web-api.html
[44] https://osisoft.my.site.com/PISquare/s/topic/0TO1I000000OPxsWAG/data-export
[45] https://docs.aveva.com/bundle/pi-web-api-reference/page/help/getting-started.html
[46] https://stackoverflow.com/questions/76889468/how-to-make-the-width-and-height-bigger-in-csv-output-file
[47] https://www.atlassian.com/data/sql/export-to-csv-from-psql
[48] https://stackoverflow.com/questions/65516932/how-to-thin-a-large-csv-file-to-extract-its-salient-features
[49] https://stackoverflow.com/questions/43471469/raspberry-pi-writing-csv-python
[50] https://proytek.com/pi4/PI%20ProcessBook%203_2%20User%20Guide.pdf
[51] https://stackoverflow.com/questions/27931169/convert-wide-format-csv-to-long-format-csv-using-python
[52] https://osicdn.blob.core.windows.net/learningcontent/pdfs/2019%20PI%20World%20PI%20Integrator%20for%20Business%20Analytics%20In-depth%20Tutorial.pdf
[53] https://www.youtube.com/watch?v=L4QzUk90UmA
[54] https://stackoverflow.com/questions/60171225/how-to-transpose-csv-data-from-a-wide-format-to-long-dataset-using-python
[55] https://www.predictiveindex.com/learn/support/download-assessment-data-csv-report/
[56] https://stackoverflow.com/questions/67505537/add-timestamp-to-file-name-during-saving-data-frame-in-csv
[57] https://support.seeq.com/kb/R61/cloud/import-csv-files
[58] https://docs.mixpanel.com/docs/export-methods
[59] https://www.reddit.com/r/learnprogramming/comments/vdxdaj/is_anyone_familiar_with_pi_by_osisoft/
[60] https://stackoverflow.com/questions/8092380/export-pcap-data-to-csv-timestamp-bytes-uplink-downlink-extra-info
[61] http://cdn.osisoft.com/learningcontent/pdfs/BuildingPISystemAssetsWorkbook.pdf
[62] https://stackoverflow.com/questions/73457842/osisoft-piconfig-export-csv-into-regular-pandas-df
[63] https://osicdn.blob.core.windows.net/learningcontent/pdfs/Data%20Entry%20with%20PI%20Manual%20Logger.pdf
[64] https://docs.tibco.com/pub/bwpluginpi/6.5.0/doc/pdf/TIB_bwpluginpi_6.5.0_user_guide.pdf
[65] https://www.youtube.com/watch?v=ClGTy_lkoTg
[66] https://stackoverflow.com/questions/29246005/powershell-export-csv-with-no-header-row
[67] https://stackoverflow.com/questions/1355876/export-table-to-file-with-column-headers-column-names-using-the-bcp-utility-an
[68] https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/export-csv?view=powershell-7.5
[69] https://www.youtube.com/watch?v=SC4gXuylItg
[70] https://github.com/imubit/pi-pbook-data-extractor
[71] https://docs.aveva.com/bundle/pi-vision/page/1010132.html
[72] https://pisquare.osisoft.com/s/question/0D51I00004UHjUFSA1/the-fastest-way-to-extract-large-amount-of-data-to-csv-files
[73] https://archive.guide.highbyte.com/kb/how_to/how_to_connections/pi_system_data_input/
[74] https://docs.aveva.com/bundle/asset-information-management/page/993987.html
[75] https://docs.aveva.com/bundle/insight/page/601018.html
[76] https://docs.aveva.com/bundle/asset-information-management/page/1042392.html
[77] https://stackoverflow.com/questions/31882074/timestamp-is-missing-when-i-export-dataset-to-csv-file
[78] https://forum.inductiveautomation.com/t/question-about-date-formatting-in-perspective-exports/97038
[79] https://docs.influxdata.com/influxdb/v2/reference/key-concepts/data-elements/
[80] https://stackoverflow.com/questions/47039299/insert-timestamp-into-filename-when-exporting-into-csv
[81] https://pisquare.osisoft.com/s/question/0D51I00004UHmivSAD/export-pi-tags-with-latest-timestamp-and-value
[82] https://support.seeq.com/kb/R62/cloud/exporting-signals-and-conditions-to-osisoft-pi
[83] https://stackoverflow.com/questions/44253026/how-to-read-csv-values-in-a-timestamp-range-in-pandas
[84] https://stackoverflow.com/questions/804118/best-timestamp-format-for-csv-excel
[85] https://pisquare.osisoft.com/s/question/0D51I00004UHhbwSAD/writing-multiple-pi-tags-and-values-to-the-pi-server
[86] https://docs.aveva.com/bundle/pi-connector-for-ufl/page/1010411.html
[87] https://pisquare.osisoft.com/s/question/0D51I00004UHhpBSAT/easy-to-use-script-to-extract-timeseries-tag-data-into-csv
[88] https://pisquare.osisoft.com/s/Blog-Detail/a8r1I000000GvJ4QAK/machine-learning-pipeline-1-importing-pi-system-data-into-python
[89] https://pisquare.osisoft.com/s/question/0D51I00004UHhESSA1/pi-data-in-excel-how-to-export-in-csv-format
[90] https://pisquare.osisoft.com/s/question/0D58W0000BYC3QESQ1/data-inconsistency-between-pi-vision-display-and-exported-csv-and-pi-web-api?nocache=https%3A%2F%2Fpisquare.osisoft.com%2Fs%2Fquestion%2F0D58W0000BYC3QESQ1%2Fdata-inconsistency-between-pi-vision-display-and-exported-csv-and-pi-web-api
[91] https://docs.aveva.com/bundle/insight/page/634485.html
[92] https://dreamreport.net/blog/writing-dream-report-data-to-historians/
[93] https://docs.aveva.com/bundle/pi-server-s-da-admin/page/1022262.html
[94] https://pisquare.osisoft.com/0D51I00004UHhQ5SAL
[95] https://osisoft.my.site.com/PISquare/s/Blog-Detail/a8r1I000000Gv8JQAS/timestamps-and-the-powershell-tools-for-the-pi-system
[96] https://pisquare.osisoft.com/s/question/0D51I00004UHnXXSA1/i-need-to-export-data-to-csv-file-with-proper-time-interval
[97] https://osisoft.my.site.com/PISquare/s/question/0D51I00004UHkgvSAD/export-trend-data-with-vba-from-processbook-to-a-csv
[98] https://docs.aveva.com/bundle/pi-server-s-da-reference/page/1022136.html
[99] https://docs.aveva.com/bundle/sp-appserver/page/248985.html
[100] https://docs.aveva.com/bundle/sp-omi-awc/page/53016.html
[101] https://pisquare.osisoft.com/s/question/0D51I00004UHllxSAD/pi-smt-feature-request-add-edit-pi-tag-to-current-values
[102] https://docs.aveva.com/bundle/pi-server-f-da-smt/page/1019383.html
[103] https://pisquare.osisoft.com/s/question/0D58W00007ADcugSAD/exporting-pi-tagspi-point-not-pi-tag-data-to-csv
[104] https://manualzz.com/doc/6918673/pi-interface-for-modbus-ethernet-plc
[105] https://duckdb.org/docs/data/csv/overview.html
[106] https://www.sql-workbench.eu/manual/command-export.html
[107] https://pisquare.osisoft.com/s/question/0D51I00004UHmtCSAT/how-to-export-the-archive-data-to-csv
[108] https://www.airslate.com/how-to/online-forms/218406-how-do-i-access-pi-data-archive
[109] https://osisoft.my.site.com/PISquare/s/topic/0TO1I000000OG95WAG/data-archive
[110] https://pisquare.osisoft.com/s/question/0D51I00004UHewrSAD/export-data-from-pi-archives-to-excel-files
[111] https://stackoverflow.com/questions/71864839/how-to-export-list-as-csv
[112] https://stackoverflow.com/questions/35900659/is-it-possible-to-include-column-names-in-the-csv-with-a-copy-into-statement-in
[113] https://community.cisco.com/t5/network-management/export-device-list-in-pi-2-0/td-p/2535570
[114] https://docs.aveva.com/bundle/intouch-hmi/page/315405.html
[115] https://docs.aveva.com/bundle/intouch-hmi/page/315602.html
[116] https://docs.aveva.com/bundle/batch-management/page/304275.html
[117] https://pisquare.osisoft.com/s/question/0D51I00004UHlxdSAD/find-a-timestamp-when-tag-was-equal-to-value-in-pi-process-book-2012
[118] https://docs.aveva.com/bundle/pi-server-f-da-smt/page/1020233.html
[119] https://osisoft.my.site.com/PISquare/s/question/0D58W0000CsaOVlSQM/does-anyone-know-if-there-is-a-pi-datalink-function-to-return-a-timestamp-when-a-historian-point-equals-a-certain-value?nocache=https%3A%2F%2Fosisoft.my.site.com%2FPISquare%2Fs%2Fquestion%2F0D58W0000CsaOVlSQM%2Fdoes-anyone-know-if-there-is-a-pi-datalink-function-to-return-a-timestamp-when-a-historian-point-equals-a-certain-value

---
Answer from Perplexity: pplx.ai/share