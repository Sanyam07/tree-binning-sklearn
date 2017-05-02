WITH dat AS (
    select * from (
        select COALESCE(ibe9350_9350,12::varchar) AS f1, 
            COALESCE(ibe9351_9351,12::varchar) AS f2,
            COALESCE(AP003318_Purchase_a_Smartphone_Mobile_Phone_V2_rank_base_20_RBGNR002,11::varchar) AS f3,
            COALESCE(AP004325_Prop_Not_Majmdcl_Ins_rank_base_10_AP004325,8::varchar) AS f4,
            COALESCE(AP000604_Pay_Bills_by_Automatically_Charging_it_to_a_Credit_Card_Fin_V2_rank_base_20_GFLG1027,8::varchar) AS f5,
            COALESCE(AP001442_Total_Any_Investments_Acquired_or_added_to_in_last_12_months_rank_base_20_BFLG2293,7::varchar) AS f6,
            COALESCE(AP000572_Banking_And_Financial_Services_Methods_Used_Banking_Online_rank_base_20_FFLGC998,8::varchar) AS f7,
            COALESCE(AP001286_Total_Liquid_Investible_Assets_Fin_dollar_ap001286::!int,1116) AS f8,
            COALESCE(AP001287_Retirement_Assets_Fin_dollar_ap001287::!int,791) AS f9,
            1 AS label
        from marketing.datamart_lead_ACXIOM_ENRICHED as dml
        join marketing.DATAMART_LEAD_SALE_ATTRIB as attrib
            on dml.datamart_lead_id = attrib.datamart_lead_id
        join acxiom.prospect_2016_q3_scored as acx
            on dml.MAINTENANCE_KEY_10050 = acx.MAINTENANCE_KEY_10050
        where attrib.datamart_sale_id IS NOT NULL
            and attrib.best_match = 't'  
    ) as pos
    UNION ALL
    select * from (
        select COALESCE(ibe9350_9350,12::varchar) AS f1, 
            COALESCE(ibe9351_9351,12::varchar) AS f2,
            COALESCE(AP003318_Purchase_a_Smartphone_Mobile_Phone_V2_rank_base_20_RBGNR002,11::varchar) AS f3,
            COALESCE(AP004325_Prop_Not_Majmdcl_Ins_rank_base_10_AP004325,8::varchar) AS f4,
            COALESCE(AP000604_Pay_Bills_by_Automatically_Charging_it_to_a_Credit_Card_Fin_V2_rank_base_20_GFLG1027,8::varchar) AS f5,
            COALESCE(AP001442_Total_Any_Investments_Acquired_or_added_to_in_last_12_months_rank_base_20_BFLG2293,7::varchar) AS f6,
            COALESCE(AP000572_Banking_And_Financial_Services_Methods_Used_Banking_Online_rank_base_20_FFLGC998,8::varchar) AS f7,
            COALESCE(AP001286_Total_Liquid_Investible_Assets_Fin_dollar_ap001286::!int,1116) AS f8,
            COALESCE(AP001287_Retirement_Assets_Fin_dollar_ap001287::!int,791) AS f9,
            0 AS label
        from marketing.datamart_lead_ACXIOM_ENRICHED as dml
        left join marketing.DATAMART_LEAD_SALE_ATTRIB as attrib
            ON dml.datamart_lead_id = attrib.datamart_lead_id
        join acxiom.prospect_2016_q3_scored as acx
            on dml.MAINTENANCE_KEY_10050 = acx.MAINTENANCE_KEY_10050
        where attrib.datamart_sale_id IS NULL
        limit 6482
    ) as neg
)
SELECT *
FROM dat
WHERE f1 != '' and f2 != '' and f3 != '' and f4 != '' 
    and f5 != '' and f6 != '' and f7 != '';