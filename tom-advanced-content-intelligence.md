            # Calculate performance for this pattern
            if 'performance' in df.columns:
                pattern_performance = np.mean(df.iloc[top_content_indices]['performance'])
            else:
                pattern_performance = 0.0
                
            patterns.append({
                'id': f"{niche}_text_pattern_{i}",
                'type': 'text',
                'keywords': pattern_words,
                'examples': examples,
                'performance': pattern_performance
            })
            
        return patterns
        
    def _extract_structure_patterns(self, df, n_patterns):
        """Extract patterns from content structure"""
        # Convert structure to numeric features
        features = []
        
        for struct in df['content_structure']:
            # Example structure: {'intro_type': 'question', 'sections': 3, 'has_list': True}
            feature = []
            
            # Extract common structural elements (simplified)
            feature.append(1 if struct.get('has_hook', False) else 0)
            feature.append(struct.get('sections', 0) / 10)  # Normalize
            feature.append(1 if struct.get('has_list', False) else 0)
            feature.append(1 if struct.get('has_question', False) else 0)
            feature.append(1 if struct.get('has_cta', False) else 0)
            feature.append(struct.get('media_count', 0) / 5)  # Normalize
            
            # Intro types (one-hot encoding)
            intro_types = ['question', 'statistic', 'story', 'controversial', 'direct']
            for intro in intro_types:
                feature.append(1 if struct.get('intro_type') == intro else 0)
                
            features.append(feature)
            
        # Cluster structures
        kmeans = KMeans(n_clusters=n_patterns, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        # Extract patterns
        patterns = []
        for i in range(n_patterns):
            # Get content in this cluster
            cluster_indices = np.where(clusters == i)[0]
            cluster_content = df.iloc[cluster_indices]
            
            if len(cluster_content) == 0:
                continue
                
            # Get most common structure elements
            structures = [s for s in cluster_content['content_structure']]
            
            # Calculate cluster performance
            if 'performance' in df.columns:
                pattern_performance = cluster_content['performance'].mean()
            else:
                pattern_performance = 0.0
                
            # Find most common structure components
            example_structures = structures[:min(5, len(structures))]
            
            patterns.append({
                'id': f"{niche}_structure_pattern_{i}",
                'type': 'structure',
                'elements': self._get_common_elements(structures),
                'examples': cluster_content['content_id'].tolist()[:5],
                'example_structures': example_structures,
                'performance': pattern_performance
            })
            
        return patterns
        
    def _extract_meta_patterns(self, df, n_patterns):
        """Extract patterns from content metadata"""
        # Use whatever columns are available
        usable_columns = [c for c in df.columns if c not in ['content_id', 'niche', 'performance']]
        
        if not usable_columns:
            return []
            
        # Group by combinations of metadata
        patterns = []
        
        # This is simplified - real implementation would use more sophisticated techniques
        # For this example, we'll just look at unique value combinations
        for col in usable_columns:
            value_performances = {}
            
            for value in df[col].unique():
                # Get content with this value
                content_subset = df[df[col] == value]
                
                if len(content_subset) < 3:
                    continue
                    
                # Calculate average performance
                if 'performance' in df.columns:
                    avg_performance = content_subset['performance'].mean()
                else:
                    avg_performance = 0.0
                    
                value_performances[value] = {
                    'count': len(content_subset),
                    'performance': avg_performance,
                    'examples': content_subset['content_id'].tolist()[:5]
                }
                
            # Sort by performance
            sorted_values = sorted(value_performances.items(), 
                                key=lambda x: x[1]['performance'], 
                                reverse=True)
            
            # Take top patterns
            for i, (value, data) in enumerate(sorted_values[:n_patterns]):
                patterns.append({
                    'id': f"{niche}_meta_pattern_{col}_{i}",
                    'type': 'metadata',
                    'field': col,
                    'value': value,
                    'count': data['count'],
                    'examples': data['examples'],
                    'performance': data['performance']
                })
                
        # Sort by performance and take top n_patterns
        patterns.sort(key=lambda x: x['performance'], reverse=True)
        return patterns[:n_patterns]
        
    def _get_common_elements(self, structures):
        """Extract common elements from a list of structures"""
        # This is a simplified approach - real implementation would be more sophisticated
        elements = {}
        
        for struct in structures:
            for key, value in struct.items():
                if key not in elements:
                    elements[key] = []
                elements[key].append(value)
                
        # Find most common value for each element
        common_elements = {}
        for key, values in elements.items():
            if isinstance(values[0], bool):
                # For boolean values, use majority
                common_elements[key] = sum(values) > len(values) / 2
            elif isinstance(values[0], (int, float)):
                # For numeric values, use average
                common_elements[key] = sum(values) / len(values)
            else:
                # For categorical values, use most common
                value_counts = pd.Series(values).value_counts()
                common_elements[key] = value_counts.index[0]
                
        return common_elements
        
    def identify_universal_patterns(self, min_niche_presence=2, performance_threshold=0.7):
        """Identify patterns that work well across multiple niches"""
        if len(self.content_patterns) < min_niche_presence:
            return []
            
        # Collect all patterns
        all_patterns = []
        for niche, patterns in self.content_patterns.items():
            for pattern in patterns:
                all_patterns.append({
                    **pattern,
                    'niche': niche
                })
                
        # Group similar patterns
        # This is a simplified approach - real implementation would use more sophisticated similarity
        pattern_groups = self._group_similar_patterns(all_patterns)
        
        # Filter for universal patterns
        universal_patterns = []
        
        for group in pattern_groups:
            # Check if pattern appears in multiple niches
            niches = set(p['niche'] for p in group)
            
            if len(niches) >= min_niche_presence:
                # Calculate average performance
                avg_performance = sum(p['performance'] for p in group) / len(group)
                
                # Only keep high-performing patterns
                if avg_performance >= performance_threshold:
                    # Create universal pattern entry
                    universal_pattern = {
                        'id': f"universal_pattern_{len(universal_patterns)}",
                        'type': group[0]['type'],
                        'niches': list(niches),
                        'niche_count': len(niches),
                        'performance': avg_performance,
                        'examples': {}
                    }
                    
                    # Add type-specific information
                    if group[0]['type'] == 'text':
                        # Combine keywords from all patterns
                        all_keywords = []
                        for p in group:
                            all_keywords.extend(p.get('keywords', []))
                        
                        # Count keyword frequency
                        keyword_counts = pd.Series(all_keywords).value_counts()
                        
                        # Keep most common keywords
                        universal_pattern['keywords'] = keyword_counts.head(20).index.tolist()
                        
                    elif group[0]['type'] == 'structure':
                        # Combine elements from all patterns
                        all_elements = [p.get('elements', {}) for p in group]
                        universal_pattern['elements'] = self._get_common_elements(all_elements)
                        
                    elif group[0]['type'] == 'metadata':
                        # Use the most common field
                        fields = [p.get('field', '') for p in group]
                        field_counts = pd.Series(fields).value_counts()
                        universal_pattern['field'] = field_counts.index[0]
                        
                        # And the most common value for that field
                        values = [p.get('value', '') for p in group if p.get('field') == universal_pattern['field']]
                        if values:
                            value_counts = pd.Series(values).value_counts()
                            universal_pattern['value'] = value_counts.index[0]
                            
                    # Add examples from each niche
                    for p in group:
                        if p['niche'] not in universal_pattern['examples'] and 'examples' in p:
                            universal_pattern['examples'][p['niche']] = p['examples']
                            
                    universal_patterns.append(universal_pattern)
                    
        # Sort by performance and niche coverage
        universal_patterns.sort(key=lambda x: (x['niche_count'], x['performance']), reverse=True)
        
        self.universal_patterns = universal_patterns
        return universal_patterns
        
    def _group_similar_patterns(self, patterns):
        """Group similar patterns together"""
        # This is a simplified approach - real implementation would use more sophisticated similarity
        
        # Group by type first
        type_groups = {}
        for pattern in patterns:
            pattern_type = pattern.get('type', 'unknown')
            if pattern_type not in type_groups:
                type_groups[pattern_type] = []
            type_groups[pattern_type].append(pattern)
            
        # Process each type group
        all_groups = []
        
        for pattern_type, type_patterns in type_groups.items():
            if pattern_type == 'text':
                # Group by keyword similarity
                groups = self._group_by_keyword_similarity(type_patterns)
            elif pattern_type == 'structure':
                # Group by structure similarity
                groups = self._group_by_structure_similarity(type_patterns)
            elif pattern_type == 'metadata':
                # Group by field and value
                groups = self._group_by_metadata_field(type_patterns)
            else:
                # Default to each pattern in its own group
                groups = [[p] for p in type_patterns]
                
            all_groups.extend(groups)
            
        return all_groups
        
    def _group_by_keyword_similarity(self, patterns, similarity_threshold=0.3):
        """Group patterns by keyword similarity"""
        # Create sets of keywords
        pattern_keywords = []
        for p in patterns:
            keywords = set(p.get('keywords', []))
            pattern_keywords.append(keywords)
            
        # Group patterns
        groups = []
        used = set()
        
        for i, p1 in enumerate(patterns):
            if i in used:
                continue
                
            group = [p1]
            used.add(i)
            
            for j, p2 in enumerate(patterns):
                if j in used or i == j:
                    continue
                    
                # Calculate similarity
                set1 = pattern_keywords[i]
                set2 = pattern_keywords[j]
                
                if not set1 or not set2:
                    continue
                    
                intersection = len(set1.intersection(set2))
                union = len(set1.union(set2))
                
                similarity = intersection / union if union > 0 else 0
                
                if similarity >= similarity_threshold:
                    group.append(p2)
                    used.add(j)
                    
            groups.append(group)
            
        return groups
        
    def _group_by_structure_similarity(self, patterns, similarity_threshold=0.5):
        """Group patterns by structure similarity"""
        # Calculate pairwise similarities
        groups = []
        used = set()
        
        for i, p1 in enumerate(patterns):
            if i in used:
                continue
                
            group = [p1]
            used.add(i)
            
            elements1 = p1.get('elements', {})
            
            for j, p2 in enumerate(patterns):
                if j in used or i == j:
                    continue
                    
                elements2 = p2.get('elements', {})
                
                # Calculate similarity
                similarity = self._calculate_structure_similarity(elements1, elements2)
                
                if similarity >= similarity_threshold:
                    group.append(p2)
                    used.add(j)
                    
            groups.append(group)
            
        return groups
        
    def _calculate_structure_similarity(self, elements1, elements2):
        """Calculate similarity between two structure elements"""
        # Common keys
        all_keys = set(elements1.keys()).union(set(elements2.keys()))
        if not all_keys:
            return 0
            
        common_keys = set(elements1.keys()).intersection(set(elements2.keys()))
        
        # Basic similarity based on common keys
        key_similarity = len(common_keys) / len(all_keys)
        
        # Value similarity for common keys
        value_similarity = 0
        
        for key in common_keys:
            if elements1[key] == elements2[key]:
                value_similarity += 1
                
        value_similarity = value_similarity / len(common_keys) if common_keys else 0
        
        # Combine similarities
        return (key_similarity + value_similarity) / 2
        
    def _group_by_metadata_field(self, patterns):
        """Group patterns by metadata field"""
        # Group by field
        field_groups = {}
        
        for pattern in patterns:
            field = pattern.get('field', 'unknown')
            if field not in field_groups:
                field_groups[field] = []
            field_groups[field].append(pattern)
            
        # For each field, further group by value
        groups = []
        
        for field, field_patterns in field_groups.items():
            value_groups = {}
            
            for pattern in field_patterns:
                value = str(pattern.get('value', 'unknown'))
                if value not in value_groups:
                    value_groups[value] = []
                value_groups[value].append(pattern)
                
            # Add each value group
            for value_patterns in value_groups.values():
                groups.append(value_patterns)
                
        return groups
        
    def adapt_pattern_to_niche(self, pattern_id, target_niche):
        """Adapt a universal pattern to a specific niche"""
        # Find the pattern
        pattern = None
        for p in self.universal_patterns:
            if p['id'] == pattern_id:
                pattern = p
                break
                
        if not pattern:
            return None
            
        # Check if target niche exists
        if target_niche not in self.niche_data:
            return None
            
        # Adapt pattern based on type
        adaptation = {
            'original_pattern': pattern_id,
            'target_niche': target_niche,
            'adaptations': []
        }
        
        if pattern['type'] == 'text':
            adaptation = self._adapt_text_pattern(pattern, target_niche, adaptation)
        elif pattern['type'] == 'structure':
            adaptation = self._adapt_structure_pattern(pattern, target_niche, adaptation)
        elif pattern['type'] == 'metadata':
            adaptation = self._adapt_metadata_pattern(pattern, target_niche, adaptation)
            
        return adaptation
        
    def _adapt_text_pattern(self, pattern, target_niche, adaptation):
        """Adapt a text pattern to a target niche"""
        # Get niche-specific vocabulary
        niche_vocab = self._extract_niche_vocabulary(target_niche)
        
        # Adapt keywords to niche vocabulary
        adapted_keywords = []
        for keyword in pattern.get('keywords', []):
            # Check if keyword already exists in niche
            if keyword in niche_vocab:
                adapted_keywords.append(keyword)
            else:
                # Find similar terms in niche vocabulary
                similar_terms = self._find_similar_terms(keyword, niche_vocab)
                if similar_terms:
                    # Add adaptation note
                    adaptation['adaptations'].append({
                        'original': keyword,
                        'adapted': similar_terms[0],
                        'alternatives': similar_terms[1:],
                        'type': 'keyword_adaptation'
                    })
                    adapted_keywords.append(similar_terms[0])
                else:
                    # Keep original keyword
                    adapted_keywords.append(keyword)
                    
        adaptation['adapted_keywords'] = adapted_keywords
        
        # Suggest content template
        template = f"Content using pattern {pattern['id']} adapted to {target_niche}:\n\n"
        template += f"Keywords to include: {', '.join(adapted_keywords[:10])}\n\n"
        template += "Suggested structure:\n"
        template += "1. Hook using keyword combination\n"
        template += "2. Main points using pattern framework\n"
        template += "3. Niche-specific examples\n"
        template += "4. Conclusion with key takeaways\n"
        
        adaptation['content_template'] = template
        return adaptation
        
    def _adapt_structure_pattern(self, pattern, target_niche, adaptation):
        """Adapt a structure pattern to a target niche"""
        # Get niche-specific structural preferences
        niche_structures = self._extract_niche_structures(target_niche)
        
        # Original pattern elements
        original_elements = pattern.get('elements', {})
        
        # Adapt structure to niche preferences
        adapted_elements = {}
        
        for key, value in original_elements.items():
            # Check if this element has niche-specific preferences
            if key in niche_structures and niche_structures[key] != value:
                # Add adaptation note
                adaptation['adaptations'].append({
                    'element': key,
                    'original': value,
                    'adapted': niche_structures[key],
                    'type': 'structure_adaptation'
                })
                adapted_elements[key] = niche_structures[key]
            else:
                # Keep original value
                adapted_elements[key] = value
                
        adaptation['adapted_elements'] = adapted_elements
        
        # Create structure template
        template = f"Content structure using pattern {pattern['id']} adapted to {target_niche}:\n\n"
        
        for key, value in adapted_elements.items():
            template += f"{key}: {value}\n"
            
        template += "\nImplementation notes:\n"
        
        for adapt in adaptation['adaptations']:
            if adapt['type'] == 'structure_adaptation':
                template += f"- Modify {adapt['element']} from {adapt['original']} to {adapt['adapted']} for this niche\n"
                
        adaptation['structure_template'] = template
        return adaptation
        
    def _adapt_metadata_pattern(self, pattern, target_niche, adaptation):
        """Adapt a metadata pattern to a target niche"""
        # Get niche-specific metadata
        niche_metadata = self._extract_niche_metadata(target_niche)
        
        # Original pattern field and value
        field = pattern.get('field', '')
        value = pattern.get('value', '')
        
        # Check if field exists in niche
        if field in niche_metadata:
            # Check if value makes sense for niche
            if value in niche_metadata[field]:
                # Value works as-is
                adaptation['adaptations'].append({
                    'field': field,
                    'original': value,
                    'adapted': value,
                    'type': 'direct_use'
                })
            else:
                # Suggest alternative value from niche
                if niche_metadata[field]:
                    adapted_value = niche_metadata[field][0]
                    adaptation['adaptations'].append({
                        'field': field,
                        'original': value,
                        'adapted': adapted_value,
                        'type': 'value_adaptation'
                    })
                    
        adaptation['metadata_recommendations'] = {
            'field': field,
            'recommended_value': adaptation['adaptations'][-1]['adapted'] if adaptation['adaptations'] else value
        }
        
        return adaptation
        
    def _extract_niche_vocabulary(self, niche):
        """Extract vocabulary specific to a niche"""
        # This is a simplified approach - real implementation would be more sophisticated
        vocabulary = set()
        
        # Extract words from content
        if niche in self.niche_data:
            for content in self.niche_data[niche]:
                if 'content_text' in content:
                    words = content['content_text'].split()
                    vocabulary.update(words)
                    
        return vocabulary
        
    def _extract_niche_structures(self, niche):
        """Extract structural preferences specific to a niche"""
        # This is a simplified approach - real implementation would be more sophisticated
        structures = {}
        
        # Count structure elements
        if niche in self.niche_data:
            element_counts = {}
            
            for content in self.niche_data[niche]:
                if 'content_structure' in content:
                    for key, value in content['content_structure'].items():
                        if key not in element_counts:
                            element_counts[key] = {}
                            
                        value_str = str(value)
                        if value_str not in element_counts[key]:
                            element_counts[key][value_str] = 0
                            
                        element_counts[key][value_str] += 1
                        
            # Find most common value for each element
            for key, counts in element_counts.items():
                most_common = max(counts.items(), key=lambda x: x[1])
                
                # Convert back to original type
                if most_common[0] == 'True':
                    structures[key] = True
                elif most_common[0] == 'False':
                    structures[key] = False
                elif most_common[0].isdigit():
                    structures[key] = int(most_common[0])
                elif most_common[0].replace('.', '').isdigit():
                    structures[key] = float(most_common[0])
                else:
                    structures[key] = most_common[0]
                    
        return structures
        
    def _extract_niche_metadata(self, niche):
        """Extract metadata preferences specific to a niche"""
        # This is a simplified approach - real implementation would be more sophisticated
        metadata = {}
        
        # Count metadata values
        if niche in self.niche_data:
            for content in self.niche_data[niche]:
                for key, value in content.items():
                    if key not in ['content_id', 'content_text', 'content_structure', 'performance']:
                        if key not in metadata:
                            metadata[key] = []
                            
                        if value not in metadata[key]:
                            metadata[key].append(value)
                            
        return metadata
        
    def _find_similar_terms(self, term, vocabulary):
        """Find terms in vocabulary similar to the given term"""
        # This is a simplified approach - real implementation would use word embeddings
        similar_terms = []
        
        # For simplicity, just find terms that contain or are contained by the target term
        for word in vocabulary:
            if term in word or word in term:
                similar_terms.append(word)
                
        # Sort by length (prioritize closer length to original term)
        similar_terms.sort(key=lambda x: abs(len(x) - len(term)))
        
        return similar_terms[:5]  # Return top 5 similar terms
```

**Key Applications**:
- Identify universal content patterns that work across seemingly unrelated niches
- Adapt successful formulas from other domains to create fresh content approaches
- Apply psychological triggers that have proven effectiveness in multiple contexts
- Create content that feels innovative but is based on validated engagement patterns

### 4. Content Polarization Optimizer

**Core Concept**: Create content with optimally calibrated controversy levels to maximize engagement while managing brand risk.

**Implementation**:
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class ContentPolarizationOptimizer:
    def __init__(self):
        self.content_data = []
        self.polarization_metrics = {}
        self.brand_risk_thresholds = {}
        self.optimal_polarization = None
        
    def add_content_data(self, content_id, metrics):
        """Add content performance data with engagement polarity metrics"""
        self.content_data.append({
            'content_id': content_id,
            'timestamp': metrics.get('timestamp', pd.Timestamp.now()),
            'polarization_score': metrics.get('polarization_score', 0),
            'positive_reactions': metrics.get('positive_reactions', 0),
            'negative_reactions': metrics.get('negative_reactions', 0),
            'neutral_reactions': metrics.get('neutral_reactions', 0),
            'total_engagement': metrics.get('total_engagement', 0),
            'conversion_rate': metrics.get('conversion_rate', 0),
            'brand_risk_score': metrics.get('brand_risk_score', 0)
        })
        
    def set_brand_risk_thresholds(self, thresholds):
        """Set brand risk thresholds"""
        self.brand_risk_thresholds = thresholds
        
    def calculate_polarization_metrics(self):
        """Calculate polarization metrics for all content"""
        if not self.content_data:
            return {}
            
        df = pd.DataFrame(self.content_data)
        
        # Calculate polarization score if not provided
        if 'polarization_score' not in df.columns or df['polarization_score'].isna().all():
            df['total_reactions'] = df['positive_reactions'] + df['negative_reactions'] + df['neutral_reactions']
            
            # Avoid division by zero
            df['total_reactions'] = df['total_reactions'].replace(0, 1)
            
            # Calculate polarization as ratio of (positive + negative) to total, ignoring neutral
            df['polarization_score'] = (df['positive_reactions'] + df['negative_reactions']) / df['total_reactions']
            
        # Calculate positive/negative ratio
        df['pos_neg_ratio'] = df['positive_reactions'] / df['negative_reactions'].replace(0, 0.001)
        
        # Calculate engagement density (engagement per unit of polarization)
        df['engagement_density'] = df['total_engagement'] / (df['polarization_score'] + 0.001)
        
        # Calculate risk-adjusted engagement
        df['risk_adjusted_engagement'] = df['total_engagement'] / (df['brand_risk_score'] + 1)
        
        # Calculate polarization efficiency
        df['polarization_efficiency'] = df['total_engagement'] / (df['polarization_score'] + 0.1) / (df['brand_risk_score'] + 1)
        
        # Store metrics
        self.polarization_metrics = {
            'mean_polarization': df['polarization_score'].mean(),
            'median_polarization': df['polarization_score'].median(),
            'max_engagement_polarization': df.loc[df['total_engagement'].idxmax(), 'polarization_score'],
            'max_conversion_polarization': df.loc[df['conversion_rate'].idxmax(), 'polarization_score'],
            'optimal_range': {
                'lower': df['polarization_score'].quantile(0.75) - 0.1,
                'upper': df['polarization_score'].quantile(0.75) + 0.1
            },
            'engagement_by_polarization': self._calculate_engagement_by_polarization(df)
        }
        
        return self.polarization_metrics
        
    def _calculate_engagement_by_polarization(self, df):
        """Calculate average engagement by polarization level"""
        # Create polarization bins
        bins = np.linspace(0, 1, 11)  # 0, 0.1, 0.2, ..., 1.0
        labels = [f"{round(bins[i], 1)}-{round(bins[i+1], 1)}" for i in range(len(bins)-1)]
        
        df['polarization_bin'] = pd.cut(df['polarization_score'], bins=bins, labels=labels)
        
        # Calculate average engagement for each bin
        engagement_by_bin = df.groupby('polarization_bin')['total_engagement'].mean().to_dict()
        conversion_by_bin = df.groupby('polarization_bin')['conversion_rate'].mean().to_dict()
        risk_by_bin = df.groupby('polarization_bin')['brand_risk_score'].mean().to_dict()
        
        return {
            'engagement': engagement_by_bin,
            'conversion': conversion_by_bin,
            'risk': risk_by_bin
        }
        
    def find_optimal_polarization(self, optimization_target='engagement'):
        """Find the optimal polarization level based on target metric"""
        if not self.polarization_metrics:
            self.calculate_polarization_metrics()
            
        if not self.polarization_metrics:
            return None
            
        # Convert metrics to DataFrame for analysis
        df = pd.DataFrame(self.content_data)
        
        # Define optimization function based on target
        if optimization_target == 'engagement':
            target_column = 'total_engagement'
        elif optimization_target == 'conversion':
            target_column = 'conversion_rate'
        elif optimization_target == 'risk_adjusted':
            target_column = 'risk_adjusted_engagement'
        else:
            target_column = 'total_engagement'
            
        # Create polarization bins
        bins = np.linspace(0, 1, 21)  # More granular bins: 0, 0.05, 0.1, ..., 1.0
        labels = [f"{round(bins[i], 2)}-{round(bins[i+1], 2)}" for i in range(len(bins)-1)]
        
        df['polarization_bin'] = pd.cut(df['polarization_score'], bins=bins, labels=labels)
        
        # Calculate average target metric for each bin
        performance_by_bin = df.groupby('polarization_bin')[target_column].mean()
        
        # Find bin with highest performance
        if len(performance_by_bin) > 0:
            best_bin = performance_by_bin.idxmax()
            
            # Extract optimal range
            bin_parts = best_bin.split('-')
            optimal_min = float(bin_parts[0])
            optimal_max = float(bin_parts[1])
            
            # Check brand risk at this polarization level
            risk_at_optimal = df[df['polarization_bin'] == best_bin]['brand_risk_score'].mean()
            
            # Apply brand risk constraints if thresholds are set
            if self.brand_risk_thresholds and 'max_acceptable' in self.brand_risk_thresholds:
                max_risk = self.brand_risk_thresholds['max_acceptable']
                
                if risk_at_optimal > max_risk:
                    # Find alternative with acceptable risk
                    acceptable_bins = df.groupby('polarization_bin').agg({
                        target_column: 'mean',
                        'brand_risk_score': 'mean'
                    })
                    
                    acceptable_bins = acceptable_bins[acceptable_bins['brand_risk_score'] <= max_risk]
                    
                    if not acceptable_bins.empty:
                        best_bin = acceptable_bins[target_column].idxmax()
                        
                        # Update optimal range
                        bin_parts = best_bin.split('-')
                        optimal_min = float(bin_parts[0])
                        optimal_max = float(bin_parts[1])
                        risk_at_optimal = acceptable_bins.loc[best_bin, 'brand_risk_score']
            
            self.optimal_polarization = {
                'min': optimal_min,
                'max': optimal_max,
                'target': (optimal_min + optimal_max) / 2,
                'performance': performance_by_bin[best_bin],
                'risk_score': risk_at_optimal,
                'optimization_target': optimization_target
            }
            
            return self.optimal_polarization
        
        return None
        
    def predict_metrics(self, polarization_score):
        """Predict metrics for a given polarization score"""
        if not self.content_data:
            return None
            
        df = pd.DataFrame(self.content_data)
        
        # Find content with similar polarization
        similar_content = df[(df['polarization_score'] >= polarization_score - 0.1) & 
                           (df['polarization_score'] <= polarization_score + 0.1)]
        
        if len(similar_content) < 3:
            # Not enough similar content, use broader range
            similar_content = df[(df['polarization_score'] >= polarization_score - 0.2) & 
                               (df['polarization_score'] <= polarization_score + 0.2)]
            
        if len(similar_content) == 0:
            return None
            
        # Calculate predicted metrics
        predictions = {
            'expected_engagement': similar_content['total_engagement'].mean(),
            'expected_conversion': similar_content['conversion_rate'].mean(),
            'expected_risk': similar_content['brand_risk_score'].mean(),
            'positive_ratio': similar_content['positive_reactions'].sum() / 
                            (similar_content['positive_reactions'].sum() + 
                             similar_content['negative_reactions'].sum() + 0.001),
            'sample_size': len(similar_content)
        }
        
        return predictions
        
    def optimize_content_polarization(self, content_concept, current_polarization=None):
        """Optimize polarization for a content concept"""
        if not self.optimal_polarization:
            self.find_optimal_polarization()
            
        if not self.optimal_polarization:
            return None
            
        # Start with optimal target
        target_polarization = self.optimal_polarization['target']
        
        # Adjustments based on content concept
        # This is a simplified approach - real implementation would be more sophisticated
        adjustments = []
        
        # If current polarization provided, calculate needed change
        if current_polarization is not None:
            change_needed = target_polarization - current_polarization
            
            if abs(change_needed) < 0.05:
                adjustment_direction = "maintain current polarization"
            elif change_needed > 0:
                adjustment_direction = "increase polarization"
            else:
                adjustment_direction = "decrease polarization"
                
            adjustments.append({
                'type': 'baseline_adjustment',
                'from': current_polarization,
                'to': target_polarization,
                'change': change_needed,
                'direction': adjustment_direction
            })
            
        # Predict metrics at target polarization
        predicted_metrics = self.predict_metrics(target_polarization)
        
        # Generate specific techniques to adjust polarization
        techniques = self._generate_polarization_techniques(
            target_polarization, 
            current_polarization,
            content_concept
        )
        
        return {
            'optimal_polarization': target_polarization,
            'range': {
                'min': self.optimal_polarization['min'],
                'max': self.optimal_polarization['max']
            },
            'adjustments': adjustments,
            'predicted_metrics': predicted_metrics,
            'techniques': techniques,
            'brand_risk': {
                'predicted': predicted_metrics['expected_risk'] if predicted_metrics else None,
                'threshold': self.brand_risk_thresholds.get('max_acceptable', None)
            }
        }
        
    def _generate_polarization_techniques(self, target, current=None, concept=None):
        """Generate specific techniques to achieve target polarization"""
        techniques = []
        
        # Determine direction
        if current is not None:
            if target > current:
                direction = "increase"
            elif target < current:
                direction = "decrease"
            else:
                direction = "maintain"
        else:
            direction = "set"
            
        # Generate techniques based on direction
        if direction == "increase":
            techniques = [
                {
                    'technique': 'Present contrasting viewpoints',
                    'description': 'Include perspectives from different sides of the issue',
                    'impact': 'Moderate increase in polarization'
                },
                {
                    'technique': 'Highlight surprising statistics',
                    'description': 'Feature data points that challenge common assumptions',
                    'impact': 'Mild increase in engagement polarity'
                },
                {
                    'technique': 'Ask provocative questions',
                    'description': 'Pose questions that make people reconsider their position',
                    'impact': 'Significant increase in comment engagement'
                },
                {
                    'technique': 'Take a clear stance',
                    'description': 'Express a definitive position rather than neutral observation',
                    'impact': 'Substantial increase in polarization'
                }
            ]
        elif direction == "decrease":
            techniques = [
                {
                    'technique': 'Focus on common ground',
                    'description': 'Emphasize areas of agreement between different perspectives',
                    'impact': 'Moderate decrease in polarization'
                },
                {
                    'technique': 'Use more neutral language',
                    'description': 'Replace emotionally charged words with more objective terms',
                    'impact': 'Significant decrease in reaction polarity'
                },
                {
                    'technique': 'Present balanced evidence',
                    'description': 'Provide equal support for multiple viewpoints',
                    'impact': 'Substantial decrease in polarization'
                },
                {
                    'technique': 'Focus on practical solutions',
                    'description': 'Emphasize actionable approaches rather than philosophical debates',
                    'impact': 'Mild decrease in controversial reactions'
                }
            ]
        else:  # maintain or set
            techniques = [
                {
                    'technique': 'Balanced perspective with clear thesis',
                    'description': 'Present multiple viewpoints while maintaining a clear position',
                    'impact': 'Controlled level of engagement polarity'
                },
                {
                    'technique': 'Strategic use of contrasting examples',
                    'description': 'Include just enough contrast to drive engagement without overwhelming',
                    'impact': 'Stable polarization at target level'
                },
                {
                    'technique': 'Measured emotional appeals',
                    'description': 'Include emotional elements without overshadowing rational content',
                    'impact': 'Maintained balance of reaction types'
                }
            ]
            
        # Add risk management techniques
        techniques.append({
            'technique': 'Brand protection framework',
            'description': 'Include disclaimer or framing that separates personal views from brand position',
            'impact': 'Reduced brand risk while maintaining engagement'
        })
        
        return techniques
```

**Key Applications**:
- Identify optimal controversy level for different content types and objectives
- Create engagement-driving content that stays within brand safety parameters
- Strategically position content to generate discussion without alienating audience
- Increase content performance by optimizing the ratio of positive to negative reactions

### 5. Psychological Velocity Tracking

**Core Concept**: Track and optimize early engagement velocity as a predictor of viral potential, enabling rapid resource allocation to emerging opportunities.

**Implementation**:
```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

class PsychologicalVelocityTracker:
    def __init__(self):
        self.content_data = {}
        self.velocity_metrics = {}
        self.velocity_patterns = {}
        self.viral_predictions = {}
        
    def add_content_timeseries(self, content_id, timestamps, metrics):
        """Add time-series engagement data for content"""
        if content_id not in self.content_data:
            self.content_data[content_id] = []
            
        # Combine timestamps and metrics into time-series data
        for i, timestamp in enumerate(timestamps):
            if i < len(metrics):
                self.content_data[content_id].append({
                    'timestamp': timestamp,
                    **metrics[i]
                })
                
        # Sort by timestamp
        self.content_data[content_id].sort(key=lambda x: x['timestamp'])
        
    def calculate_velocity_metrics(self, content_id, early_window_minutes=60):
        """Calculate velocity metrics for specific content"""
        if content_id not in self.content_data or not self.content_data[content_id]:
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(self.content_data[content_id])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Calculate time since first data point in minutes
        start_time = df.index.min()
        df['minutes_since_start'] = (df.index - start_time).total_seconds() / 60
        
        # Define early window
        early_df = df[df['minutes_since_start'] <= early_window_minutes]
        
        if len(early_df) < 3:  # Need at least 3 data points for meaningful analysis
            return None
            
        # Calculate metrics for different engagement types
        metrics = {}
        
        for col in df.columns:
            if col not in ['minutes_since_start'] and df[col].dtype in [np.int64, np.float64]:
                # Calculate early velocity (rate of change)
                if len(early_df) >= 3:
                    X = early_df['minutes_since_start'].values.reshape(-1, 1)
                    y = early_df[col].values
                    
                    try:
                        # Add constant for intercept
                        X = sm.add_constant(X)
                        
                        # Fit linear model
                        model = sm.OLS(y, X).fit()
                        
                        # Get slope (velocity)
                        velocity = model.params[1]
                        
                        # Get acceleration (change in velocity)
                        if len(early_df) >= 4:
                            # Calculate first differences (velocity at each point)
                            early_df[f'{col}_velocity'] = early_df[col].diff() / early_df['minutes_since_start'].diff()
                            
                            # Calculate acceleration (change in velocity)
                            X_accel = early_df['minutes_since_start'].values[1:].reshape(-1, 1)
                            y_accel = early_df[f'{col}_velocity'].values[1:]
                            
                            if len(y_accel) >= 2:
                                X_accel = sm.add_constant(X_accel)
                                accel_model = sm.OLS(y_accel, X_accel).fit()
                                acceleration = accel_model.params[1]
                            else:
                                acceleration = 0
                        else:
                            acceleration = 0
                            
                        # Calculate R-squared to measure linearity
                        r_squared = model.rsquared
                        
                        metrics[col] = {
                            'early_velocity': velocity,
                            'acceleration': acceleration,
                            'r_squared': r_squared,
                            'total_early': early_df[col].max(),
                            'final_value': df[col].iloc[-1]
                        }
                    except:
                        # Skip if model fails
                        continue
                        
        # Calculate pattern metrics
        pattern_metrics = self._analyze_engagement_pattern(df, early_df)
        
        # Combine all metrics
        all_metrics = {
            'content_id': content_id,
            'early_window_minutes': early_window_minutes,
            'data_points': len(df),
            'early_data_points': len(early_df),
            'total_duration_minutes': df['minutes_since_start'].max(),
            'engagement_metrics': metrics,
            'pattern_metrics': pattern_metrics
        }
        
        # Store metrics
        self.velocity_metrics[content_id] = all_metrics
        
        return all_metrics
        
    def _analyze_engagement_pattern(self, df, early_df):
        """Analyze engagement pattern for viral prediction"""
        pattern_metrics = {}
        
        # Check for acceleration pattern (exponential growth)
        for col in df.columns:
            if col not in ['minutes_since_start'] and df[col].dtype in [np.int64, np.float64]:
                # Check if we have enough data
                if len(early_df) >= 5:
                    # Extract x and y
                    x = early_df['minutes_since_start'].values
                    y = early_df[col].values
                    
                    # Skip if all values are the same
                    if np.std(y) == 0:
                        continue
                        
                    try:
                        # Compare linear vs exponential model fit
                        # Linear model
                        X_lin = sm.add_constant(x)
                        lin_model = sm.OLS(y, X_lin).fit()
                        lin_aic = lin_model.aic
                        
                        # Exponential model (log transform y)
                        # Add small constant to avoid log(0)
                        y_log = np.log(y + 1)
                        exp_model = sm.OLS(y_log, X_lin).fit()
                        exp_aic = exp_model.aic
                        
                        # Compare models (lower AIC is better)
                        is_exponential = exp_aic < lin_aic
                        
                        # Calculate growth factor for exponential model
                        # growth_factor = e^slope
                        if is_exponential:
                            growth_factor = np.exp(exp_model.params[1])
                        else:
                            growth_factor = 1.0
                            
                        pattern_metrics[f'{col}_pattern'] = {
                            'is_exponential': is_exponential,
                            'growth_factor': growth_factor,
                            'exp_model_fit': exp_model.rsquared,
                            'lin_model_fit': lin_model.rsquared
                        }
                    except:
                        # Skip if model fails
                        continue
                        
        # Check for viral signature patterns
        # 1. Early commenting/sharing ratio
        if 'comments' in df.columns and 'views' in df.columns and len(early_df) > 0:
            early_comment_rate = early_df['comments'].max() / (early_df['views'].max() + 1)
            pattern_metrics['early_comment_rate'] = early_comment_rate
            
        # 2. Share to like ratio
        if 'shares' in df.columns and 'likes' in df.columns and early_df['likes'].max() > 0:
            share_like_ratio = early_df['shares'].max() / (early_df['likes'].max() + 1)
            pattern_metrics['share_like_ratio'] = share_like_ratio
            
        # 3. Engagement velocity consistency
        if 'engagement' in df.columns and len(early_df) >= 3:
            # Calculate velocity at each time point
            early_df['engagement_velocity'] = early_df['engagement'].diff() / early_df['minutes_since_start'].diff()
            
            # Calculate coefficient of variation (lower means more consistent)
            velocity_mean = early_df['engagement_velocity'].mean()
            velocity_std = early_df['engagement_velocity'].std()
            
            if velocity_mean > 0:
                velocity_cv = velocity_std / velocity_mean
                pattern_metrics['velocity_consistency'] = 1 / (1 + velocity_cv)  # Higher is more consistent
                
        return pattern_metrics
        
    def analyze_viral_patterns(self, min_content_items=5):
        """Analyze patterns across content to identify viral indicators"""
        if len(self.velocity_metrics) < min_content_items:
            return None
            
        # Extract metrics for analysis
        metrics_list = []
        
        for content_id, metrics in self.velocity_metrics.items():
            # Get known final performance
            final_metrics = {}
            
            for metric_name, metric_data in metrics['engagement_metrics'].items():
                final_metrics[f'final_{metric_name}'] = metric_data['final_value']
                final_metrics[f'early_velocity_{metric_name}'] = metric_data['early_velocity']
                final_metrics[f'acceleration_{metric_name}'] = metric_data['acceleration']
                
            # Add pattern metrics
            for pattern_name, pattern_value in metrics['pattern_metrics'].items():
                if isinstance(pattern_value, dict):
                    for k, v in pattern_value.items():
                        final_metrics[f'pattern_{pattern_name}_{k}'] = v
                else:
                    final_metrics[f'pattern_{pattern_name}'] = pattern_value
                    
            # Add to list
            metrics_list.append({
                'content_id': content_id,
                **final_metrics
            })
            
        # Convert to DataFrame
        df = pd.DataFrame(metrics_list)
        
        if len(df) == 0:
            return None
            
        # Identify correlations between early metrics and final performance
        correlations = {}
        
        # Identify viral content (top 10% performers)
        if 'final_engagement' in df.columns:
            viral_threshold = df['final_engagement'].quantile(0.9)
            df['is_viral'] = df['final_engagement'] >= viral_threshold
            
            # Calculate correlation between early metrics and viral status
            for col in df.columns:
                if col.startswith('early_velocity_') or col.startswith('acceleration_') or col.startswith('pattern_'):
                    try:
                        point_biserial = stats.pointbiserialr(df[col], df['is_viral'])
                        correlations[col] = {
                            'correlation': point_biserial.correlation,
                            'p_value': point_biserial.pvalue,
                            'significant': point_biserial.pvalue < 0.05
                        }
                    except:
                        continue
                        
        # Identify top viral indicators
        significant_indicators = {k: v for k, v in correlations.items() if v['significant']}
        sorted_indicators = sorted(significant_indicators.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
        
        top_indicators = []
        for indicator, stats in sorted_indicators[:5]:  # Top 5 indicators
            top_indicators.append({
                'metric': indicator,
                'correlation': stats['correlation'],
                'p_value': stats['p_value']
            })
            
        # Create viral threshold values
        viral_thresholds = {}
        
        for indicator in [i['metric'] for i in top_indicators]:
            if indicator in df.columns and df['is_viral'].sum() > 0:
                viral_values = df[df['is_viral']][indicator]
                if len(viral_values) > 0:
                    viral_thresholds[indicator] = viral_values.min()
                    
        # Store patterns
        self.velocity_patterns = {
            'top_indicators': top_indicators,
            'viral_thresholds': viral_thresholds,
            'viral_ratio': df['is_viral'].mean() if 'is_viral' in df.columns else 0
        }
        
        return self.velocity_patterns
        
    def predict_viral_potential(self, content_id):
        """Predict viral potential for content based on early velocity"""
        # Check if we have velocity metrics for this content
        if content_id not in self.velocity_metrics:
            return None
            
        # Check if we have identified patterns
        if not self.velocity_patterns or 'top_indicators' not in self.velocity_patterns:
            self.analyze_viral_patterns()
            
        if not self.velocity_patterns or 'top_indicators' not in self.velocity_patterns:
            return None
            
        # Get content metrics
        metrics = self.velocity_metrics[content_id]
        
        # Extract values for top indicators
        indicator_values = {}
        for indicator in [i['metric'] for i in self.velocity_patterns['top_indicators']]:
            parts = indicator.split('_')
            
            if indicator.startswith('early_velocity_'):
                metric_name = '_'.join(parts[2:])
                if metric_name in metrics['engagement_metrics']:
                    indicator_values[indicator] = metrics['engagement_metrics'][metric_name]['early_velocity']
                    
            elif indicator.startswith('acceleration_'):
                metric_name = '_'.join(parts[1:])
                if metric_name in metrics['engagement_metrics']:
                    indicator_values[indicator] = metrics['engagement_metrics'][metric_name]['acceleration']
                    
            elif indicator.startswith('pattern_'):
                pattern_parts = parts[1:]
                pattern_name = '_'.join(pattern_parts)
                
                # Check if it's a nested pattern metric
                if len(pattern_parts) > 1 and '_'.join(pattern_parts[:-1]) in metrics['pattern_metrics']:
                    base_pattern = '_'.join(pattern_parts[:-1])
                    sub_metric = pattern_parts[-1]
                    
                    if isinstance(metrics['pattern_metrics'][base_pattern], dict) and sub_metric in metrics['pattern_metrics'][base_pattern]:
                        indicator_values[indicator] = metrics['pattern_metrics'][base_pattern][sub_metric]
                elif pattern_name in metrics['pattern_metrics']:
                    indicator_values[indicator] = metrics['pattern_metrics'][pattern_name]
                    
        # Calculate viral score based on thresholds
        viral_score = 0
        max_score = 0
        
        for indicator, value in indicator_values.items():
            if indicator in self.velocity_patterns['viral_thresholds']:
                threshold = self.velocity_patterns['viral_thresholds'][indicator]
                max_score += 1
                
                if value >= threshold:
                    viral_score += 1
                    
        # Normalize score
        if max_score > 0:
            normalized_score = viral_score / max_score
        else:
            normalized_score = 0
            
        # Calculate probability adjustment based on base rate
        base_rate = self.velocity_patterns['viral_ratio']
        
        # Simple Bayesian-inspired adjustment
        if base_rate > 0:
            # P(viral | indicators) ≈ P(indicators | viral) * P(viral) / P(indicators)
            # We approximate P(indicators | viral) with normalized_score
            # P(viral) is the base_rate
            # We approximate P(indicators) with base_rate*normalized_score + (1-base_rate)*0.5
            p_indicators_given_viral = normalized_score
            p_viral = base_rate
            p_indicators = base_rate * normalized_score + (1 - base_rate) * 0.1  # Assuming false positive rate of 0.1
            
            bayesian_probability = min(1.0, p_indicators_given_viral * p_viral / max(0.01, p_indicators))
        else:
            bayesian_probability = normalized_score
            
        # Create prediction
        prediction = {
            'content_id': content_id,
            'viral_score': normalized_score,
            'viral_probability': bayesian_probability,
            'indicator_values': indicator_values,
            'thresholds_met': viral_score,
            'total_thresholds': max_score
        }
        
        # Add projections based on current velocity
        projections = {}
        
        for metric_name, metric_data in metrics['engagement_metrics'].items():
            velocity = metric_data['early_velocity']
            acceleration = metric_data['acceleration']
            current_value = metric_data['final_value']
            
            # Project 24-hour value based on current velocity and acceleration
            # Simple projection: current + velocity*time + 0.5*acceleration*time^2
            minutes_remaining = max(0, 24*60 - metrics['total_duration_minutes'])
            
            projection = current_value
            
            if minutes_remaining > 0:
                # Linear component
                projection += velocity * minutes_remaining
                
                # Acceleration component (limit to avoid extreme values)
                accel_component = 0.5 * min(abs(acceleration), abs(velocity)*0.1) * (minutes_remaining ** 2)
                if acceleration < 0:
                    accel_component = -accel_component
                    
                projection += accel_component
                
                # Ensure non-negative
                projection = max(0, projection)
                
            projections[metric_name] = {
                'current': current_value,
                'projected_24h': projection,
                'multiplier': projection / current_value if current_value > 0 else 0
            }
            
        prediction['projections'] = projections
        
        # Store prediction
        self.viral_predictions[content_id] = prediction
        
        return prediction
        
    def get_resource_allocation_recommendation(self, content_id):
        """Get resource allocation recommendation based on viral potential"""
        # Check if we have a prediction
        if content_id not in self.viral_predictions:
            prediction = self.predict_viral_potential(content_id)
            if not prediction:
                return None
        else:
            prediction = self.viral_predictions[content_id]
            
        # Define allocation tiers
        allocation_tiers = [
            {
                'threshold': 0.9,
                'label': 'Extreme Viral Potential',
                'allocation_multiplier': 10.0,
                'urgency': 'Immediate - All Available Resources',
                'actions': [
                    'Deploy cross-platform promotion immediately',
                    'Allocate paid promotion budget',
                    'Create derivative content to amplify reach',
                    'Prepare for high-volume community management',
                    'Alert executive team about potential brand moment'
                ]
            },
            {
                'threshold': 0.8,
                'label': 'Very High Viral Potential',
                'allocation_multiplier': 5.0,
                'urgency': 'Urgent - Priority Resources',
                'actions': [
                    'Promote across all owned channels',
                    'Consider limited paid promotion',
                    'Prepare follow-up content',
                    'Increase community management resources',
                    'Notify leadership team'
                ]
            },
            {
                'threshold': 0.6,
                'label': 'High Viral Potential',
                'allocation_multiplier': 2.5,
                'urgency': 'High Priority',
                'actions': [
                    'Promote on main social channels',
                    'Develop related content',
                    'Assign dedicated community manager',
                    'Consider cross-promotion opportunities',
                    'Monitor performance closely'
                ]
            },
            {
                'threshold': 0.4,
                'label': 'Above Average Potential',
                'allocation_multiplier': 1.5,
                'urgency': 'Medium Priority',
                'actions': [
                    'Promote on relevant channels',
                    'Engage with early commenters',
                    'Monitor for increased traction',
                    'Consider light optimizations',
                    'Tag for content repurposing'
                ]
            },
            {
                'threshold': 0.0,
                'label': 'Standard Potential',
                'allocation_multiplier': 1.0,
                'urgency': 'Normal Priority',
                'actions': [
                    'Standard promotion protocol',
                    'Regular engagement levels',
                    'Normal monitoring',
                    'No special resource allocation needed'
                ]
            }
        ]
        
        # Determine allocation tier
        tier = None
        for t in allocation_tiers:
            if prediction['viral_probability'] >= t['threshold']:
                tier = t
                break
                
        # Default to standard if no tier matched
        if not tier:
            tier = allocation_tiers[-1]
            
        # Create recommendation
        recommendation = {
            'content_id': content_id,
            'viral_probability': prediction['viral_probability'],
            'tier': tier['label'],
            'urgency': tier['urgency'],
            'resource_multiplier': tier['allocation_multiplier'],
            'recommended_actions': tier['actions'],
            'projections': prediction['projections']
        }
        
        # Add time sensitivity
        if 'engagement' in prediction['projections']:
            proj = prediction['projections']['engagement']
            if proj['multiplier'] > 5:
                recommendation['time_sensitivity'] = 'Extremely Time-Sensitive'
            elif proj['multiplier'] > 3:
                recommendation['time_sensitivity'] = 'Highly Time-Sensitive'
            elif proj['multiplier'] > 2:
                recommendation['time_sensitivity'] = 'Moderately Time-Sensitive'
            else:
                recommendation['time_sensitivity'] = 'Standard Time Sensitivity'
                
        return recommendation
```

**Key Applications**:
- Identify content with viral potential in the critical early window
- Predict performance trajectories based on early engagement signals
- Allocate resources to emerging opportunities before competitors notice
- Develop patterns that consistently generate strong initial velocity

## Implementation Strategy

To implement these advanced systems, follow this approach:

1. **Start with Social Arbitrage** - It provides immediate value with lower implementation complexity
2. **Add Psychological Velocity Tracking** - This gives you early warning capabilities for viral opportunities
3. **Implement Trend Intersection Detector** - Use this to identify high-potential content ideas
4. **Add Pattern Borrowing System** - Once you have data from multiple niches, implement this for creative advantage
5. **Finally, implement Content Polarization Optimizer** - This is most valuable once you have engagement data to analyze

By implementing these systems in this order, you'll create a powerful content intelligence engine that significantly outperforms competitors and maximizes your social media empire's revenue potential.
# TOM v1.0.0 — Advanced Content Intelligence Systems

## Advanced Content Strategies for Maximum Impact

### 1. Social Arbitrage System

**Core Concept**: Leverage performance differentials across platforms to identify validated content opportunities with reduced risk.

**Implementation**:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SocialArbitrageSystem:
    def __init__(self, platforms=None):
        self.platforms = platforms or []
        self.platform_data = {platform: [] for platform in self.platforms}
        self.opportunity_scores = {}
        self.migration_strategies = {}
        
    def add_platform(self, platform_name):
        """Add a new platform to monitor"""
        if platform_name not in self.platforms:
            self.platforms.append(platform_name)
            self.platform_data[platform_name] = []
            
    def add_content_performance(self, platform, content_id, metrics):
        """Add performance data for content on a specific platform"""
        if platform not in self.platform_data:
            self.add_platform(platform)
            
        self.platform_data[platform].append({
            'content_id': content_id,
            'timestamp': metrics.get('timestamp', pd.Timestamp.now()),
            **metrics
        })
        
    def identify_arbitrage_opportunities(self, performance_metric='engagement_rate', 
                                        min_data_points=5, lookback_days=30):
        """Identify content performing well on some platforms but not yet exploited on others"""
        opportunities = []
        
        # Ensure we have enough data
        if len(self.platforms) < 2:
            return opportunities
            
        # Convert platform data to DataFrames
        platform_dfs = {}
        for platform, data in self.platform_data.items():
            if len(data) < min_data_points:
                continue
                
            df = pd.DataFrame(data)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Filter to recent data
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback_days)
                df = df[df['timestamp'] >= cutoff]
                
            platform_dfs[platform] = df
            
        if len(platform_dfs) < 2:
            return opportunities
            
        # Calculate performance statistics for each platform
        platform_stats = {}
        for platform, df in platform_dfs.items():
            if performance_metric not in df.columns:
                continue
                
            platform_stats[platform] = {
                'mean': df[performance_metric].mean(),
                'median': df[performance_metric].median(),
                'std': df[performance_metric].std(),
                'max': df[performance_metric].max(),
                'min': df[performance_metric].min()
            }
            
        # Normalize performance across platforms
        all_content = {}
        scaler = MinMaxScaler()
        
        for platform, df in platform_dfs.items():
            if performance_metric not in df.columns:
                continue
                
            for _, row in df.iterrows():
                content_id = row['content_id']
                
                if content_id not in all_content:
                    all_content[content_id] = {'platforms': {}}
                    
                # Add normalized performance 
                # (using z-score relative to platform averages for fair comparison)
                if platform_stats[platform]['std'] > 0:
                    z_score = (row[performance_metric] - platform_stats[platform]['mean']) / platform_stats[platform]['std']
                else:
                    z_score = 0
                    
                all_content[content_id]['platforms'][platform] = {
                    'raw_performance': row[performance_metric],
                    'z_score': z_score,
                    'timestamp': row.get('timestamp', None),
                    'metrics': {k: v for k, v in row.items() 
                              if k not in ['content_id', 'timestamp']}
                }
                
        # Identify opportunities (content performing well on some platforms but not on others)
        for content_id, data in all_content.items():
            if len(data['platforms']) < 2:
                # Need presence on multiple platforms for comparison
                continue
                
            # Sort platforms by performance
            sorted_platforms = sorted(
                data['platforms'].keys(),
                key=lambda p: data['platforms'][p]['z_score'],
                reverse=True
            )
            
            best_platform = sorted_platforms[0]
            best_score = data['platforms'][best_platform]['z_score']
            
            # Only consider content performing significantly above average on at least one platform
            if best_score < 1.0:  # At least 1 standard deviation above mean
                continue
                
            # Check for platforms where this content isn't present
            missing_platforms = [p for p in self.platforms if p not in data['platforms']]
            
            # Check for platforms where performance is significantly lower
            underperforming = []
            for platform in sorted_platforms[1:]:
                score_diff = best_score - data['platforms'][platform]['z_score']
                if score_diff > 1.5:  # Significant difference in performance
                    underperforming.append({
                        'platform': platform,
                        'current_score': data['platforms'][platform]['z_score'],
                        'potential_lift': score_diff
                    })
                    
            if missing_platforms or underperforming:
                # Calculate opportunity score (combination of performance and potential)
                opportunity_score = best_score * (1 + 0.5 * len(missing_platforms) + 
                                              0.3 * sum(u['potential_lift'] for u in underperforming))
                
                opportunities.append({
                    'content_id': content_id,
                    'best_platform': best_platform,
                    'best_score': best_score,
                    'missing_platforms': missing_platforms,
                    'underperforming_platforms': underperforming,
                    'opportunity_score': opportunity_score,
                    'content_metrics': data['platforms'][best_platform]['metrics']
                })
                
        # Sort by opportunity score
        opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)
        
        # Store opportunity scores
        self.opportunity_scores = {op['content_id']: op['opportunity_score'] for op in opportunities}
        
        return opportunities
        
    def generate_migration_strategy(self, content_id, target_platform):
        """Generate strategy for migrating successful content to a new platform"""
        if content_id not in self.opportunity_scores:
            return None
            
        # Find original content across platforms
        content_data = {}
        best_platform = None
        best_score = -float('inf')
        
        for platform, data in self.platform_data.items():
            for item in data:
                if item['content_id'] == content_id:
                    if platform not in content_data:
                        content_data[platform] = []
                    content_data[platform].append(item)
                    
                    # Track best performing platform
                    if 'engagement_rate' in item and (best_platform is None or 
                                                    item['engagement_rate'] > best_score):
                        best_platform = platform
                        best_score = item['engagement_rate']
        
        if not best_platform or target_platform not in self.platforms:
            return None
            
        # Analyze platform differences
        platform_differences = self._analyze_platform_differences(best_platform, target_platform)
        
        # Generate adaptation strategy
        strategy = {
            'content_id': content_id,
            'source_platform': best_platform,
            'target_platform': target_platform,
            'adaptations': [
                {
                    'element': 'format',
                    'change': platform_differences.get('format_change', 'Maintain same format')
                },
                {
                    'element': 'duration',
                    'change': platform_differences.get('duration_change', 'Maintain same duration')
                },
                {
                    'element': 'style',
                    'change': platform_differences.get('style_change', 'Adapt to platform norms')
                },
                {
                    'element': 'call_to_action',
                    'change': platform_differences.get('cta_change', 'Platform-appropriate CTA')
                }
            ],
            'opportunity_score': self.opportunity_scores[content_id]
        }
        
        # Store migration strategy
        if content_id not in self.migration_strategies:
            self.migration_strategies[content_id] = {}
        self.migration_strategies[content_id][target_platform] = strategy
        
        return strategy
        
    def _analyze_platform_differences(self, source_platform, target_platform):
        """Analyze differences between platforms to inform adaptation strategy"""
        # This would be implemented with platform-specific knowledge
        # Simplified example:
        platform_characteristics = {
            'tiktok': {'vertical': True, 'short_form': True, 'audio_important': True},
            'instagram': {'vertical': True, 'visual_quality': 'high', 'carousel': True},
            'youtube': {'horizontal': True, 'longer_form': True, 'search_driven': True},
            'twitter': {'text_heavy': True, 'conversation': True, 'concise': True},
            'linkedin': {'professional': True, 'text_and_image': True, 'business': True}
        }
        
        # Default values if platforms not in our dictionary
        source_chars = platform_characteristics.get(source_platform.lower(), {})
        target_chars = platform_characteristics.get(target_platform.lower(), {})
        
        differences = {}
        
        # Format adaptation
        if source_chars.get('vertical', False) and target_chars.get('horizontal', False):
            differences['format_change'] = 'Convert from vertical to horizontal format'
        elif source_chars.get('horizontal', False) and target_chars.get('vertical', False):
            differences['format_change'] = 'Convert from horizontal to vertical format'
            
        # Duration adaptation
        if source_chars.get('short_form', False) and target_chars.get('longer_form', False):
            differences['duration_change'] = 'Expand content with additional context and details'
        elif source_chars.get('longer_form', False) and target_chars.get('short_form', False):
            differences['duration_change'] = 'Condense to key points and moments'
            
        # Style adaptation
        if target_chars.get('professional', False):
            differences['style_change'] = 'Adapt tone to be more professional and industry-focused'
        elif target_chars.get('text_heavy', False):
            differences['style_change'] = 'Emphasize textual elements and conversation hooks'
            
        # CTA adaptation
        if target_chars.get('business', False):
            differences['cta_change'] = 'Business-oriented call to action'
        elif target_chars.get('conversation', False):
            differences['cta_change'] = 'Question-based call to action to drive engagement'
            
        return differences
```

**Key Applications**:
- Identify content that's performing 2-3x better on one platform and adapt for others
- Discover content from smaller platforms that has proven engagement before wider release
- Create risk-reduced content strategy by focusing on pre-validated concepts
- Build cross-platform synergies that competitors typically miss

### 2. Trend Intersection Detector

**Core Concept**: Identify the convergence points of multiple rising trends where viral potential is exponentially higher.

**Implementation**:
```python
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class TrendIntersectionDetector:
    def __init__(self):
        self.trends = {}
        self.trend_vectors = {}
        self.intersections = []
        self.trend_graph = nx.Graph()
        
    def add_trend(self, trend_name, trend_data):
        """Add or update a trend with its associated data"""
        self.trends[trend_name] = {
            'name': trend_name,
            'data': trend_data,
            'timestamps': trend_data.get('timestamps', []),
            'momentum': trend_data.get('momentum', 0),
            'keywords': trend_data.get('keywords', []),
            'categories': trend_data.get('categories', []),
            'audience': trend_data.get('audience', [])
        }
        
        # Create vector representation of the trend
        self._vectorize_trend(trend_name)
        
        # Update trend graph
        self.trend_graph.add_node(trend_name, **self.trends[trend_name])
        
    def _vectorize_trend(self, trend_name):
        """Create vector representation of a trend for similarity comparison"""
        if trend_name not in self.trends:
            return
            
        trend = self.trends[trend_name]
        
        # This is a simplified vector representation
        # Real implementation would use more sophisticated embeddings
        vector = []
        
        # Add keyword embeddings (simplified)
        for keyword in trend['keywords']:
            # This would use word embeddings in real implementation
            # Here we just use a hash-based approach for demonstration
            hash_val = hash(keyword) % 10000
            vector.append(hash_val / 10000)  # Normalize to 0-1
            
        # Add category indicators
        categories = ['technology', 'entertainment', 'business', 'health', 
                     'sports', 'politics', 'fashion', 'food', 'travel', 'education']
        
        for category in categories:
            vector.append(1.0 if category in trend['categories'] else 0.0)
            
        # Add audience indicators
        audiences = ['gen_z', 'millennials', 'gen_x', 'boomers', 
                    'male', 'female', 'professional', 'student']
        
        for audience in audiences:
            vector.append(1.0 if audience in trend['audience'] else 0.0)
            
        # Add momentum
        vector.append(min(1.0, trend['momentum'] / 100.0))  # Normalize to 0-1
        
        # Store vector
        self.trend_vectors[trend_name] = np.array(vector)
        
    def detect_intersections(self, similarity_threshold=0.3, min_momentum=20, 
                           max_intersection_size=4):
        """Detect intersections between trends based on similarity and momentum"""
        if len(self.trends) < 2:
            return []
            
        # Filter trends by momentum
        active_trends = [t for t, data in self.trends.items() 
                       if data['momentum'] >= min_momentum]
        
        if len(active_trends) < 2:
            return []
            
        # Create similarity matrix
        vectors = [self.trend_vectors[t] for t in active_trends]
        similarity_matrix = cosine_similarity(vectors)
        
        # Create graph from similarity matrix
        G = nx.Graph()
        for i, trend1 in enumerate(active_trends):
            G.add_node(trend1, **self.trends[trend1])
            
            for j, trend2 in enumerate(active_trends):
                if i < j and similarity_matrix[i, j] >= similarity_threshold:
                    G.add_edge(trend1, trend2, weight=similarity_matrix[i, j])
        
        # Find communities (potential intersections)
        communities = list(nx.community.greedy_modularity_communities(G))
        
        # Process communities to find viable intersections
        intersections = []
        for i, community in enumerate(communities):
            if len(community) < 2 or len(community) > max_intersection_size:
                continue
                
            # Calculate intersection metrics
            trends_in_intersection = [self.trends[t] for t in community]
            total_momentum = sum(t['momentum'] for t in trends_in_intersection)
            avg_momentum = total_momentum / len(community)
            
            # Combined keywords from all trends in intersection
            all_keywords = set()
            for trend in trends_in_intersection:
                all_keywords.update(trend['keywords'])
                
            # Common categories and audiences
            categories = set(trends_in_intersection[0]['categories'])
            audiences = set(trends_in_intersection[0]['audience'])
            
            for trend in trends_in_intersection[1:]:
                categories.intersection_update(trend['categories'])
                audiences.intersection_update(trend['audience'])
                
            # Calculate intersection score
            # Higher for: higher momentum, more connected trends, more common elements
            density = nx.density(G.subgraph(community))
            intersection_score = avg_momentum * density * (1 + len(categories) * 0.2 + len(audiences) * 0.1)
            
            intersections.append({
                'id': f"intersection_{i}",
                'trends': list(community),
                'trend_names': [self.trends[t]['name'] for t in community],
                'size': len(community),
                'total_momentum': total_momentum,
                'avg_momentum': avg_momentum,
                'common_categories': list(categories),
                'common_audiences': list(audiences),
                'keywords': list(all_keywords),
                'density': density,
                'intersection_score': intersection_score
            })
            
        # Sort by intersection score
        intersections.sort(key=lambda x: x['intersection_score'], reverse=True)
        
        self.intersections = intersections
        return intersections
        
    def generate_content_ideas(self, intersection_id, num_ideas=5):
        """Generate content ideas for a specific trend intersection"""
        # Find the intersection
        intersection = None
        for inter in self.intersections:
            if inter['id'] == intersection_id:
                intersection = inter
                break
                
        if not intersection:
            return []
            
        # This is a simplified idea generation approach
        # Real implementation would use more sophisticated techniques
        
        ideas = []
        trends = [self.trends[t] for t in intersection['trends']]
        
        # Idea 1: Comparison/contrast
        if len(trends) >= 2:
            ideas.append({
                'type': 'comparison',
                'title': f"How {trends[0]['name']} and {trends[1]['name']} Are Changing {'/'.join(intersection['common_categories'])}",
                'format': 'explainer',
                'hook': f"The unexpected connection between {trends[0]['name']} and {trends[1]['name']} that everyone's missing",
                'keywords': intersection['keywords'][:5]
            })
            
        # Idea 2: Comprehensive guide
        ideas.append({
            'type': 'guide',
            'title': f"The Complete Guide to {' + '.join(t['name'] for t in trends[:2])}",
            'format': 'tutorial',
            'hook': f"Master the intersection of {' and '.join(t['name'] for t in trends)} before your competition",
            'keywords': intersection['keywords'][:5]
        })
        
        # Idea 3: Future prediction
        ideas.append({
            'type': 'prediction',
            'title': f"Why {' + '.join(t['name'] for t in trends[:2])} Will Transform {random.choice(intersection['common_categories'])}",
            'format': 'analysis',
            'hook': f"The future belongs to those who understand this powerful trend combination",
            'keywords': intersection['keywords'][:5]
        })
        
        # Idea 4: Case study
        ideas.append({
            'type': 'case_study',
            'title': f"How Innovative Brands Are Leveraging {' and '.join(t['name'] for t in trends[:2])}",
            'format': 'story',
            'hook': f"Learn from the pioneers who are already capitalizing on this trend intersection",
            'keywords': intersection['keywords'][:5]
        })
        
        # Idea 5: Controversy/debate
        ideas.append({
            'type': 'controversy',
            'title': f"The {trends[0]['name']} vs {trends[1]['name']} Debate: What You Need to Know",
            'format': 'opinion',
            'hook': f"Why experts are divided on how these major trends will interact",
            'keywords': intersection['keywords'][:5]
        })
        
        return ideas[:num_ideas]
        
    def visualize_trend_network(self):
        """Create visualization data for trend network"""
        if len(self.trend_vectors) < 2:
            return None
            
        # Create graph for visualization
        G = nx.Graph()
        
        # Add nodes
        for trend_name, trend_data in self.trends.items():
            G.add_node(trend_name, 
                     size=trend_data['momentum'],
                     group=trend_data['categories'][0] if trend_data['categories'] else 'other',
                     label=trend_name)
            
        # Add edges based on similarity
        trends = list(self.trends.keys())
        vectors = [self.trend_vectors[t] for t in trends]
        similarity_matrix = cosine_similarity(vectors)
        
        for i in range(len(trends)):
            for j in range(i+1, len(trends)):
                similarity = similarity_matrix[i, j]
                if similarity >= 0.3:  # Threshold for showing connection
                    G.add_edge(trends[i], trends[j], weight=similarity)
        
        # Convert to visualization format
        nodes = [{'id': node, 
                'size': data.get('size', 10), 
                'group': data.get('group', 'other'),
                'label': data.get('label', node)} 
               for node, data in G.nodes(data=True)]
        
        edges = [{'source': u, 
                'target': v, 
                'value': data.get('weight', 1)} 
               for u, v, data in G.edges(data=True)]
        
        return {
            'nodes': nodes,
            'links': edges
        }
```

**Key Applications**:
- Identify high-potential content opportunities where multiple trends converge
- Create content positioned at the center of multiple audience interests
- Develop "trend bridge" content that connects different audience segments
- Generate ideas with significantly higher viral potential than single-trend content

### 3. Pattern Borrowing System

**Core Concept**: Analyze viral patterns across unrelated niches and adapt them to your content for unique approaches competitors won't recognize.

**Implementation**:
```python
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
import random

class PatternBorrowingSystem:
    def __init__(self):
        self.content_patterns = {}
        self.niche_data = {}
        self.universal_patterns = []
        self.pattern_effectiveness = {}
        
    def add_niche_data(self, niche_name, content_data):
        """Add content performance data for a specific niche"""
        if niche_name not in self.niche_data:
            self.niche_data[niche_name] = []
            
        self.niche_data[niche_name].extend(content_data)
        
    def extract_patterns(self, min_patterns=3, max_patterns=10):
        """Extract content patterns from each niche"""
        for niche, content_list in self.niche_data.items():
            if len(content_list) < 10:  # Need sufficient data
                continue
                
            # Convert to DataFrame
            df = pd.DataFrame(content_list)
            
            # Select appropriate number of patterns based on data size
            n_patterns = min(max(min_patterns, len(df) // 20), max_patterns)
            
            # Extract patterns using NMF
            # This is a simplified approach - real implementation would be more sophisticated
            if 'content_text' in df.columns:
                patterns = self._extract_text_patterns(df, n_patterns)
            elif 'content_structure' in df.columns:
                patterns = self._extract_structure_patterns(df, n_patterns)
            else:
                patterns = self._extract_meta_patterns(df, n_patterns)
                
            # Store patterns with effectiveness metrics
            self.content_patterns[niche] = patterns
            
    def _extract_text_patterns(self, df, n_patterns):
        """Extract patterns from content text"""
        # This would use NLP techniques in real implementation
        # Simplified version uses word presence as features
        
        # Extract top words across all content
        all_text = ' '.join(df['content_text'])
        words = all_text.split()
        word_counts = pd.Series(words).value_counts()
        top_words = word_counts.head(1000).index.tolist()
        
        # Create document-term matrix
        document_term = np.zeros((len(df), len(top_words)))
        
        for i, text in enumerate(df['content_text']):
            for j, word in enumerate(top_words):
                if word in text:
                    document_term[i, j] = 1
                    
        # Apply NMF
        model = NMF(n_components=n_patterns, random_state=42)
        W = model.fit_transform(document_term)
        H = model.components_
        
        # Extract patterns
        patterns = []
        for i in range(n_patterns):
            # Get top words for this pattern
            pattern_scores = H[i]
            top_indices = pattern_scores.argsort()[-20:][::-1]
            pattern_words = [top_words[j] for j in top_indices]
            
            # Find top content examples for this pattern
            content_scores = W[:, i]
            top_content_indices = content_scores.argsort()[-5:][::-1]
            examples = df.iloc[top_content_indices]['content_id'].tolist()
            
            # Calculate performance for this