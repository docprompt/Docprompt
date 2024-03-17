import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  Svg: React.ComponentType<React.ComponentProps<'svg'>>;
  description: JSX.Element;
};

const FeatureList: FeatureItem[] = [
  {
    title: 'Easy to Use',
    Svg: require('@site/static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Docprompt was designed from the ground up to get up and running quickly.
        A robust API and abstraction layer means you can get started in minutes, no matter which providers you use.
      </>
    ),
  },
  {
    title: 'Turn-key Document Analysis, Zero-Shot',
    Svg: require('@site/static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Never train a custom model again. Docprompt provides first-class support for large language models to allow zero-shot support
        for tasks including classification, segmentation, table extraction, and more.
      </>
    ),
  },
  {
    title: 'Optimized for Speed',
    Svg: require('@site/static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Extensive use of Ghostscript and PikePdf means common operations are fast and efficient.
        Batch processing means tasks like OCR can run on thousands of pages in <b>seconds.</b>
      </>
    ),
  },
];

function Feature({title, Svg, description}: FeatureItem) {
  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        <Svg className={styles.featureSvg} role="img" />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures(): JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
