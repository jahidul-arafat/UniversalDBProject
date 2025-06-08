// Fixed getDatabaseStatsEnhanced function with better error handling and null safety

async function getDatabaseStatsEnhanced() {
    if (!isConnected) {
        throw new Error('No database connected');
    }

    try {
        showLoading('Calculating comprehensive database statistics...');

        // Initialize comprehensive stats object with safe defaults
        const comprehensiveStats = {
            database_overview: {
                database_name: 'Unknown',
                database_type: 'SQLite',
                file_size_mb: 0,
                total_tables: 0,
                total_views: 0,
                total_triggers: 0,
                total_rows: 0,
                total_columns: 0,
                schema_version: 'Unknown',
                created_date: new Date().toISOString().split('T')[0],
                last_analyzed: new Date().toISOString()
            },
            table_analysis: {
                total_tables: 0,
                largest_table: null,
                smallest_table: null,
                average_rows_per_table: 0,
                median_rows_per_table: 0,
                empty_tables: 0,
                large_tables: 0,
                medium_tables: 0,
                small_tables: 0,
                table_size_distribution: {
                    empty: 0,
                    small: 0,
                    medium: 0,
                    large: 0
                }
            },
            column_analysis: {
                total_columns: 0,
                nullable_columns: 0,
                non_nullable_columns: 0,
                nullable_percentage: 0,
                most_common_type: 'N/A',
                type_distribution: {},
                average_columns_per_table: 0,
                max_columns_per_table: 0,
                min_columns_per_table: 0
            },
            index_analysis: {
                total_indexes: 0,
                user_created_indexes: 0,
                system_indexes: 0,
                unique_indexes: 0,
                composite_indexes: 0,
                single_column_indexes: 0,
                tables_with_indexes: 0,
                tables_without_indexes: 0,
                average_indexes_per_table: 0,
                index_coverage_percentage: 0,
                indexing_strategy: 'Balanced'
            },
            constraint_analysis: {
                total_constraints: 0,
                primary_key_constraints: 0,
                foreign_key_constraints: 0,
                unique_constraints: 0,
                check_constraints: 0,
                not_null_constraints: 0,
                tables_with_primary_key: 0,
                tables_without_primary_key: 0,
                primary_key_coverage: 0
            },
            relationship_analysis: {
                total_foreign_keys: 0,
                total_relationships: 0,
                tables_with_relationships: 0,
                tables_without_relationships: 0,
                referenced_tables: 0,
                relationship_density: 0,
                circular_references: 0,
                cascade_actions: {
                    cascade_deletes: 0,
                    cascade_updates: 0,
                    restrict_deletes: 0,
                    set_null_actions: 0
                }
            },
            data_quality: {
                data_quality_score: 100,
                tables_with_data: 0,
                empty_tables: 0,
                tables_with_null_issues: 0,
                data_consistency_score: 90,
                referential_integrity_score: 95,
                completeness_score: 100
            },
            performance_metrics: {
                estimated_query_performance: 'Good',
                index_utilization: 'Balanced',
                join_efficiency: 'Optimized',
                estimated_maintenance_overhead: 'Low'
            },
            storage_analysis: {
                file_size_mb: 0,
                estimated_data_size_mb: 0,
                estimated_index_size_mb: 0,
                estimated_metadata_size_mb: 0,
                average_row_size_bytes: 0,
                storage_efficiency: 'Good',
                growth_trend: 'Stable'
            },
            schema_complexity: {
                complexity_score: 10,
                complexity_level: 'Simple',
                maintainability: 'High',
                schema_depth: 1,
                interconnectedness: 0
            }
        };

        // Get basic database info with error handling
        try {
            const dbInfo = await DatabaseExplorerAPI.getDatabaseInfo();
            if (dbInfo && dbInfo.success) {
                comprehensiveStats.database_overview = {
                    ...comprehensiveStats.database_overview,
                    database_name: dbInfo.database_name || 'Unknown',
                    database_type: dbInfo.database_type || 'SQLite',
                    file_size_mb: parseFloat(dbInfo.file_size_mb) || 0,
                    total_tables: parseInt(dbInfo.total_tables) || 0,
                    total_views: parseInt(dbInfo.total_views) || 0,
                    total_triggers: parseInt(dbInfo.total_triggers) || 0,
                    total_rows: parseInt(dbInfo.total_rows) || 0,
                    schema_version: dbInfo.schema_version || 'SQLite'
                };
            }
        } catch (error) {
            console.warn('Error getting database info:', error);
        }

        // Get detailed table information with error handling
        let tables = [];
        try {
            const tablesResult = await DatabaseExplorerAPI.getTables();
            tables = (tablesResult && tablesResult.success) ? (tablesResult.tables || []) : [];
        } catch (error) {
            console.warn('Error getting tables:', error);
        }

        // Calculate table analysis
        if (tables.length > 0) {
            try {
                const tableSizes = tables.map(t => parseInt(t.rows) || 0).sort((a, b) => b - a);
                const columnCounts = tables.map(t => parseInt(t.columns) || 0);
                const totalColumns = columnCounts.reduce((sum, count) => sum + count, 0);

                comprehensiveStats.database_overview.total_columns = totalColumns;

                comprehensiveStats.table_analysis = {
                    total_tables: tables.length,
                    largest_table: tables.find(t => (t.rows || 0) === Math.max(...tableSizes)) || null,
                    smallest_table: tables.find(t => (t.rows || 0) === Math.min(...tableSizes)) || null,
                    average_rows_per_table: tableSizes.length > 0 ? Math.round(tableSizes.reduce((sum, rows) => sum + rows, 0) / tables.length) : 0,
                    median_rows_per_table: tableSizes.length > 0 ? tableSizes[Math.floor(tableSizes.length / 2)] : 0,
                    empty_tables: tables.filter(t => (t.rows || 0) === 0).length,
                    large_tables: tables.filter(t => (t.rows || 0) > 10000).length,
                    medium_tables: tables.filter(t => (t.rows || 0) > 1000 && (t.rows || 0) <= 10000).length,
                    small_tables: tables.filter(t => (t.rows || 0) > 0 && (t.rows || 0) <= 1000).length,
                    table_size_distribution: {
                        empty: tables.filter(t => (t.rows || 0) === 0).length,
                        small: tables.filter(t => (t.rows || 0) > 0 && (t.rows || 0) <= 1000).length,
                        medium: tables.filter(t => (t.rows || 0) > 1000 && (t.rows || 0) <= 10000).length,
                        large: tables.filter(t => (t.rows || 0) > 10000).length
                    }
                };
            } catch (error) {
                console.warn('Error calculating table analysis:', error);
            }
        }

        // Get schema objects for enhanced analysis with comprehensive error handling
        let indexes = [], views = [], triggers = [], constraints = [], foreignKeys = [];

        try {
            const indexesResult = await SchemaAPI.getAllIndexes().catch(() => ({ success: false, indexes: [] }));
            indexes = (indexesResult && indexesResult.success) ? (indexesResult.indexes || []) : [];
        } catch (error) {
            console.warn('Error getting indexes:', error);
        }

        try {
            const viewsResult = await SchemaAPI.getAllViews().catch(() => ({ success: false, views: [] }));
            views = (viewsResult && viewsResult.success) ? (viewsResult.views || []) : [];
        } catch (error) {
            console.warn('Error getting views:', error);
        }

        try {
            const triggersResult = await SchemaAPI.getAllTriggers().catch(() => ({ success: false, triggers: [] }));
            triggers = (triggersResult && triggersResult.success) ? (triggersResult.triggers || []) : [];
        } catch (error) {
            console.warn('Error getting triggers:', error);
        }

        try {
            const constraintsResult = await SchemaAPI.getAllConstraints().catch(() => ({ success: false, constraints: [] }));
            constraints = (constraintsResult && constraintsResult.success) ? (constraintsResult.constraints || []) : [];
        } catch (error) {
            console.warn('Error getting constraints:', error);
        }

        try {
            const foreignKeysResult = await SchemaAPI.getAllForeignKeys().catch(() => ({ success: false, foreign_keys: [] }));
            foreignKeys = (foreignKeysResult && foreignKeysResult.success) ? (foreignKeysResult.foreign_keys || []) : [];
        } catch (error) {
            console.warn('Error getting foreign keys:', error);
        }

        // Analyze indexes with safe defaults
        try {
            const userIndexes = indexes.filter(idx => idx && idx.name && !idx.name.startsWith('sqlite_'));

            comprehensiveStats.index_analysis = {
                total_indexes: indexes.length,
                user_created_indexes: userIndexes.length,
                system_indexes: indexes.length - userIndexes.length,
                unique_indexes: indexes.filter(idx => idx && idx.unique).length,
                composite_indexes: indexes.filter(idx => idx && Array.isArray(idx.columns) && idx.columns.length > 1).length,
                single_column_indexes: indexes.filter(idx => idx && Array.isArray(idx.columns) && idx.columns.length === 1).length,
                tables_with_indexes: new Set(indexes.filter(idx => idx && idx.table_name).map(idx => idx.table_name)).size,
                tables_without_indexes: Math.max(0, tables.length - new Set(indexes.filter(idx => idx && idx.table_name).map(idx => idx.table_name)).size),
                average_indexes_per_table: tables.length > 0 ? Math.round((userIndexes.length / tables.length) * 100) / 100 : 0,
                index_coverage_percentage: tables.length > 0 ? Math.round((new Set(indexes.filter(idx => idx && idx.table_name).map(idx => idx.table_name)).size / tables.length) * 100) : 0,
                indexing_strategy: userIndexes.length > tables.length ? 'Over-indexed' :
                    userIndexes.length === 0 ? 'Under-indexed' : 'Balanced'
            };
        } catch (error) {
            console.warn('Error analyzing indexes:', error);
        }

        // Analyze constraints with safe defaults
        try {
            comprehensiveStats.constraint_analysis = {
                total_constraints: constraints.length,
                primary_key_constraints: constraints.filter(c => c && c.type === 'PRIMARY KEY').length,
                foreign_key_constraints: constraints.filter(c => c && c.type === 'FOREIGN KEY').length,
                unique_constraints: constraints.filter(c => c && c.type === 'UNIQUE').length,
                check_constraints: constraints.filter(c => c && c.type === 'CHECK').length,
                not_null_constraints: constraints.filter(c => c && c.type === 'NOT NULL').length,
                tables_with_primary_key: new Set(constraints.filter(c => c && c.type === 'PRIMARY KEY' && c.table_name).map(c => c.table_name)).size,
                tables_without_primary_key: Math.max(0, tables.length - new Set(constraints.filter(c => c && c.type === 'PRIMARY KEY' && c.table_name).map(c => c.table_name)).size),
                primary_key_coverage: tables.length > 0 ? Math.round((new Set(constraints.filter(c => c && c.type === 'PRIMARY KEY' && c.table_name).map(c => c.table_name)).size / tables.length) * 100) : 0
            };
        } catch (error) {
            console.warn('Error analyzing constraints:', error);
        }

        // Analyze relationships with safe defaults
        try {
            comprehensiveStats.relationship_analysis = {
                total_foreign_keys: foreignKeys.length,
                total_relationships: foreignKeys.length,
                tables_with_relationships: new Set(foreignKeys.filter(fk => fk && fk.table_name).map(fk => fk.table_name)).size,
                tables_without_relationships: Math.max(0, tables.length - new Set(foreignKeys.filter(fk => fk && fk.table_name).map(fk => fk.table_name)).size),
                referenced_tables: new Set(foreignKeys.filter(fk => fk && fk.referenced_table).map(fk => fk.referenced_table)).size,
                relationship_density: tables.length > 1 ? Math.round((foreignKeys.length / (tables.length * (tables.length - 1))) * 10000) / 100 : 0,
                circular_references: 0, // Would need complex analysis
                cascade_actions: {
                    cascade_deletes: foreignKeys.filter(fk => fk && fk.on_delete === 'CASCADE').length,
                    cascade_updates: foreignKeys.filter(fk => fk && fk.on_update === 'CASCADE').length,
                    restrict_deletes: foreignKeys.filter(fk => fk && fk.on_delete === 'RESTRICT').length,
                    set_null_actions: foreignKeys.filter(fk => fk && fk.on_delete === 'SET NULL').length
                }
            };
        } catch (error) {
            console.warn('Error analyzing relationships:', error);
        }

        // Analyze column types and data quality with error handling
        let totalColumns = 0;
        let nullableColumns = 0;
        let columnTypes = {};

        for (const table of tables) {
            try {
                const structureResult = await DatabaseExplorerAPI.getTableStructure(table.name);
                if (structureResult && structureResult.success && structureResult.structure && structureResult.structure.columns) {
                    const columns = structureResult.structure.columns;
                    totalColumns += columns.length;
                    nullableColumns += columns.filter(col => col && col.nullable).length;

                    columns.forEach(col => {
                        if (col && col.type) {
                            const type = col.type.toUpperCase();
                            columnTypes[type] = (columnTypes[type] || 0) + 1;
                        }
                    });
                }
            } catch (error) {
                console.warn(`Error analyzing table ${table.name}:`, error);
            }
        }

        // Column analysis with safe defaults
        try {
            const sortedColumnTypes = Object.entries(columnTypes).sort(([,a], [,b]) => b - a);

            comprehensiveStats.column_analysis = {
                total_columns: totalColumns,
                nullable_columns: nullableColumns,
                non_nullable_columns: totalColumns - nullableColumns,
                nullable_percentage: totalColumns > 0 ? Math.round((nullableColumns / totalColumns) * 100) : 0,
                most_common_type: sortedColumnTypes.length > 0 ? sortedColumnTypes[0][0] : 'N/A',
                type_distribution: Object.fromEntries(sortedColumnTypes),
                average_columns_per_table: tables.length > 0 ? Math.round((totalColumns / tables.length) * 100) / 100 : 0,
                max_columns_per_table: tables.length > 0 ? Math.max(...tables.map(t => parseInt(t.columns) || 0)) : 0,
                min_columns_per_table: tables.length > 0 ? Math.min(...tables.map(t => parseInt(t.columns) || 1)) : 0
            };
        } catch (error) {
            console.warn('Error analyzing columns:', error);
        }

        // Data quality analysis with safe calculations
        try {
            const emptyTablesCount = tables.filter(t => (t.rows || 0) === 0).length;
            const tablesWithDataCount = tables.filter(t => (t.rows || 0) > 0).length;

            let qualityScore = 100;
            if (emptyTablesCount > 0) qualityScore -= emptyTablesCount * 5;
            if (comprehensiveStats.constraint_analysis.primary_key_coverage < 100) qualityScore -= 10;
            if (comprehensiveStats.index_analysis.index_coverage_percentage < 50) qualityScore -= 15;

            comprehensiveStats.data_quality = {
                data_quality_score: Math.max(0, Math.min(100, qualityScore)),
                tables_with_data: tablesWithDataCount,
                empty_tables: emptyTablesCount,
                tables_with_null_issues: Math.floor(nullableColumns * 0.1), // Estimated
                data_consistency_score: Math.max(80, 100 - emptyTablesCount * 5),
                referential_integrity_score: foreignKeys.length > 0 ? 95 : Math.max(75, 95 - emptyTablesCount * 2),
                completeness_score: totalColumns > 0 ? Math.round(((totalColumns - nullableColumns) / totalColumns) * 100) : 100
            };
        } catch (error) {
            console.warn('Error analyzing data quality:', error);
        }

        // Performance metrics with safe defaults
        try {
            const indexCoverage = comprehensiveStats.index_analysis.index_coverage_percentage;
            comprehensiveStats.performance_metrics = {
                estimated_query_performance: indexCoverage > 80 ? 'Excellent' :
                    indexCoverage > 60 ? 'Good' :
                        indexCoverage > 40 ? 'Fair' : 'Poor',
                index_utilization: comprehensiveStats.index_analysis.indexing_strategy,
                join_efficiency: foreignKeys.length > 0 ? 'Optimized' : 'Limited',
                estimated_maintenance_overhead: comprehensiveStats.index_analysis.user_created_indexes > tables.length * 3 ? 'High' :
                    comprehensiveStats.index_analysis.user_created_indexes > tables.length ? 'Medium' : 'Low'
            };
        } catch (error) {
            console.warn('Error calculating performance metrics:', error);
        }

        // Storage analysis with safe calculations
        try {
            const fileSize = comprehensiveStats.database_overview.file_size_mb;
            const totalRows = comprehensiveStats.database_overview.total_rows;
            const avgRowSize = totalRows > 0 ? (fileSize * 1024 * 1024) / totalRows : 0;

            comprehensiveStats.storage_analysis = {
                file_size_mb: fileSize,
                estimated_data_size_mb: fileSize * 0.7, // Estimated
                estimated_index_size_mb: fileSize * 0.2, // Estimated
                estimated_metadata_size_mb: fileSize * 0.1, // Estimated
                average_row_size_bytes: Math.round(avgRowSize),
                storage_efficiency: fileSize < 100 ? 'Excellent' :
                    fileSize < 500 ? 'Good' : 'Consider optimization',
                growth_trend: 'Stable' // Would need historical data
            };
        } catch (error) {
            console.warn('Error calculating storage analysis:', error);
        }

        // Schema complexity with safe calculations
        try {
            const complexityScore = Math.min(100,
                (tables.length * 2) +
                (foreignKeys.length * 3) +
                (comprehensiveStats.index_analysis.user_created_indexes * 1) +
                (constraints.length * 1) +
                (triggers.length * 4) +
                (views.length * 2)
            );

            comprehensiveStats.schema_complexity = {
                complexity_score: complexityScore,
                complexity_level: complexityScore < 20 ? 'Simple' :
                    complexityScore < 50 ? 'Moderate' :
                        complexityScore < 80 ? 'Complex' : 'Very Complex',
                maintainability: complexityScore < 60 ? 'High' :
                    complexityScore < 80 ? 'Medium' : 'Low',
                schema_depth: Math.max(1, Math.ceil(Math.log2(tables.length + 1))),
                interconnectedness: foreignKeys.length > 0 ? Math.round((foreignKeys.length / Math.max(tables.length, 1)) * 100) / 100 : 0
            };
        } catch (error) {
            console.warn('Error calculating schema complexity:', error);
        }

        return {
            success: true,
            stats: comprehensiveStats,
            generated_at: new Date().toISOString(),
            analysis_version: '2.0'
        };

    } catch (error) {
        console.error('Error calculating enhanced database statistics:', error);
        return {
            success: false,
            error: error.message,
            stats: null
        };
    }
}